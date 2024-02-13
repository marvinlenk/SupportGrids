"""
Scaling function for the logarithmic grid - x0 has to be added
"""
function expgrid(i, a, b)
  return exp(a * i + b)
end

"""
Weights for integration, not used so far
"""
function expweights(i, a, b)
  return a * exp(a * i + b)
end

function ExpMesh_params(σ::Real, sclose::Real, L::Int)
  b = log(sclose)
  a = (log(σ) - b) / (L - 1)
  return a, b
end

"""
Logarithmic mesh base
"""
mutable struct ExpMesh{T<:Real} <: AbstractNonlinearMesh{T}
  xmin::T    # Left bound
  xmax::T    # Right bound
  x0::T      # center - not necessary a point of the grid
  σ::T       # width of the grid
  L::Int     # Number of points
  L_L::Int     # Number of points on the left
  a_L::T       # Scaling parameter 1
  b_L::T       # Scaling parameter 2
  L_R::Int     # Number of points on the right
  a_R::T       # Scaling parameter 1
  b_R::T       # Scaling parameter 2

  sclose::T    # distance of closest value to x0
  
  mesh::Vector{T}
  op::NonlinearMeshOps{T}

  function ExpMesh(xmin::Real, xmax::Real, L::Int; sclose::Real=1e-9, T::Type{<:Real}=Float64,
      x0::Real=(xmax + xmin)/2)
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
      L_L = L ÷ 2
      L_R = L_L + L % 2
    end
    
    a_L, b_L = ExpMesh_params(σfactor * σ, sclose, L_L)
    a_R, b_R = ExpMesh_params(σfactor * σ, sclose, L_R)
    
    lingrid_L = UnitRange(0, L_L - 1)
    posgrid_L = expgrid.(lingrid_L, a_L, b_L)
    if L_L != 0
      posgrid_L[end] = σfactor * σ
    end
    
    lingrid_R = UnitRange(0, L_R - 1)
    posgrid_R = expgrid.(lingrid_R, a_R, b_R)
    if L_R != 0
      posgrid_R[end] = σfactor * σ 
    end
    
    mesh = [-reverse(posgrid_L); posgrid_R] .+ x0
    op = NonlinearMeshOps(mesh)
    new{T}(xmin, xmax, x0, σ, L, L_L, a_L, b_L, L_R, a_R, b_R, sclose, mesh, op)
  end
  #
  ExpMesh(::Type{T}, args...; kwargs...) where {T<:Real} = ExpMesh(args...; T, kwargs...)
end
  
"""
Update Mesh - necessary for composite meshes

not updated!
"""
function update_mesh!(exm::ExpMesh)
  @warn "Not updated update_mesh! function!"  
  exm.L % 2 == 0 ? nothing : ArgumentError("Length of logarightmic mesh must be even")
  exm.xmin = exm.x0 - exm.σ/2
  exm.xmax = exm.x0 + exm.σ/2
  exm.L2 = Int(exm.L/2)
  
  lingrid = UnitRange(0, exm.L2 - 1)
  posgrid = expgrid.(lingrid, exm.a, exm.b)
  exm.mesh = [-reverse(posgrid); posgrid] .+ exm.x0
  exm.op = NonlinearMeshOps(exm.mesh)
end

"""
    invert_mesh(exm::ExpMesh{T}, y::Real) where {T}

Get index of `y` on the mesh `exm` (not rounded - i.e. allow for interpolated values)
actually might be wrong for sclose < 1e-12 somehow
also has problems with negative values in only positive grid

See also [`ExpMesh`](@ref).
"""
function invert_mesh(exm::ExpMesh{T}, y::Real) where {T}
  @unpack L_L, a_L, b_L, L_R, a_R, b_R, x0, sclose = exm
  y = convert(T, y)
  # extrapolate exponentially between zero and first value
  if abs(y - x0) < sclose
    if y - x0 > zero(T)
      return T(L_L) + (3/2)^((y - x0)/sclose) - 0.5
    else
      return T(L_L) - (3/2)^(-(y - x0)/sclose) + 1.5
    end
  end
  # add 1 to convert from 0:L-1 unit range to 1:L index
  # loggrid is created on positive y + y0 and mirrored
  # first check positive side
  y - x0 > zero(T) && return (log(y - x0) - b_R)/ a_R + one(T) + T(L_L)
  # then negative
  return T(L_L) - (log(-y + x0) - b_L)/ a_L
end
