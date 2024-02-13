# This linearizes a Lorentzian
"""
Scaling function for the square root grid - x0 has to be added
"""
function sqrtgrid(i, a, b)
  return sqrt(abs(b / (a * i + b) - 1)) / (π*b)
end

# """
# Weights for integration, not used so far
# """
# function sqrtweights(i, a, b)
#   return nothing
# end

function SqrtMesh_params(σ::Real, γ::Real, L::Int)
  b = 1/(π * γ)
  a = (lorentzian(σ,γ) - b) / (L-1)
  return a, b
end

"""
Non-equidistant grid that samples a Lorentzian of width `γ` centered at `x0` such that the function values are equidistant.

See also [`lorentzian`](@ref).
"""
mutable struct SqrtMesh{T<:Real} <: AbstractNonlinearMesh{T}
  xmin::T    # Left bound
  xmax::T    # Right bound
  x0::T      # center - not necessary a point of the grid
  σ::Real       # width of the grid
  L::Int     # Number of points
  L_L::Int     # Number of points on the left
  a_L::Real       # Scaling parameter 1
  b_L::Real       # Scaling parameter 2
  L_R::Int     # Number of points on the right
  a_R::Real       # Scaling parameter 1
  b_R::Real       # Scaling parameter 2
  γ::Real         # Width of the Lorentzian
  
  mesh::Vector{T}
  op::NonlinearMeshOps{T}
  @doc"""
       SqrtMesh(xmin::Real, xmax::Real, L::Int; γ::Real=1, T::Type{<:Real}=Float64)
  
  Constructor of [`SqrtMesh`](@ref), where `γ` is the width of the Lorentzian.
  """
  function SqrtMesh(xmin::Real, xmax::Real, L::Int; γ::Real=1, T::Type{<:Real}=Float64,
      xdens::Real=(xmax-xmin)/2)
    x0=(xmax + xmin)/2
    σ = xmax - xmin
    σfactor = 1
    if x0 == xmin
      L_L = 0
      L_R = L
    elseif x0 == xmax
      L_L = L
      L_R = 0
    else
      σfactor = 1/2
      L_L = L ÷ 2 + 1 # remove x0 from this grid
      L_R = L_L + L % 2
    end
    
    a_L, b_L = SqrtMesh_params(σfactor * σ, γ, L_L)
    a_R, b_R = SqrtMesh_params(σfactor * σ, γ, L_R)
    
    lingrid_L = UnitRange(0, L_L - 1)
    posgrid_L = sqrtgrid.(lingrid_L, a_L, b_L)
    if L_L != 0
      posgrid_L[end] = σfactor * σ
    end
    println(posgrid_L[1:50:end])
    
    lingrid_R = UnitRange(0, L_R - 1)
    posgrid_R = sqrtgrid.(lingrid_R, a_R, b_R)
    if L_R != 0
      posgrid_R[end] = σfactor * σ
    end
    println(posgrid_R[1:50:end])
    
    mesh = [-reverse(posgrid_L)[1:end-1]; posgrid_R] .+ x0
    op = NonlinearMeshOps(mesh)
    new{T}(xmin, xmax, x0, σ, L, L_L, a_L, b_L, L_R, a_R, b_R, γ, mesh, op)
  end
  #
  @doc"""
       SqrtMesh(::Type{T}, args...; kwargs...) = SqrtMesh(args...; T, kwargs...)
  """
  SqrtMesh(::Type{T}, args...; kwargs...) where {T<:Real} = SqrtMesh(args...; T, kwargs...)
end

"""
    invert_mesh(sqm::SqrtMesh{T}, y::Real) where {T}

Get index of y on the mesh (not rounded - i.e. allow for interpolated values)

See also [`SqrtMesh`](@ref).
"""
function invert_mesh(sqm::SqrtMesh{T}, y::Real) where {T}
  @unpack L_L, a_L, b_L, a_R, b_R, x0, γ = sqm
  y = convert(T, y)
  # extrapolate exponentially between zero and first value
  if y < x0
    y_eff = x0 - y
    return L_L - (lorentzian(y_eff,γ) - b_L) / a_L
  else
    y_eff = y - x0
    return L_L - 1 + (lorentzian(y_eff,γ) - b_R) / a_R
  end
end
