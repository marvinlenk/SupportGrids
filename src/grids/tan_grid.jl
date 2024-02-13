"""
Scaling function for the logarithmic grid - x0 has to be added
"""
function tangrid(i, a, b, s)
  return tan(a * i + b) * s
end

"""
Weights for integration, not used so far
"""
function tanweights(i, a, b, s)
  return a * s / (cos(a * i + b))^2
end

function TanMesh_params(xmin::Real, xmax::Real, L::Int; xdens::Real, s::Real)
  L % 2 == 0 ? nothing : ArgumentError("Length of tangens mesh must be even")
  xmin_eff = (xmin - xdens)/s
  xmax_eff = (xmax - xdens)/s
  a = (atan(xmax_eff) - atan(xmin_eff))/(L - 1)
  b = atan(xmin_eff)
  return a, b
end

"""
Non-equidistant grid that samples the arcustangens `atan(x/s)` such that the function values are equidistant. This can provide a mapping between an infinite and a finite interval. The derivative of the arcustangens is a Lorentzian, so the point density follows a Lorentzian shape.

See also [`atan`](@ref), [`lorentzian`](@ref).
"""
mutable struct TanMesh{T<:Real} <: AbstractNonlinearMesh{T}
  xmin::T   # minimum point
  xmax::T   # maximum point
  xdens::T  # dense point
  L::Int    # Number of points
  a::T      # Scaling parameter 1
  b::T      # Scaling parameter 2
  s::T      # Scaling parameter 3
  
  mesh::Vector{T}
  op::NonlinearMeshOps{T}
  
  @doc"""
      TanMesh(xmin::Real, xmax::Real, L::Int; xdens::Real=(xmax+xmin)/2, s::Real=1,
              T::Type{<:Real}=Float64)
  
  Constructor of [`TanMesh`](@ref), where `s` determines the spread of the points. For `s→∞` the grid becomes equidistant (although `s=Inf` is not supported). Respectively, for `s≪1` the grid becomes very dense around `sclose`.
  """
  function TanMesh(xmin::Real, xmax::Real, L::Int; xdens::Real=(xmax+xmin)/2, s::Real=1, T::Type{<:Real}=Float64)
    a, b = TanMesh_params(xmin, xmax, L; xdens, s)
    lingrid = UnitRange(0, L - 1)
    mesh = tangrid.(lingrid, a, b, s) .+ xdens
    op = NonlinearMeshOps(mesh)
    new{T}(xmin, xmax, xdens, L, a, b, s, mesh, op)
  end
  #
  @doc"""
      TanMesh(::Type{T}, args...; kwargs...) where {T<:Real} = TanMesh(args...; T, kwargs...)
  """
  TanMesh(::Type{T}, args...; kwargs...) where {T<:Real} = TanMesh(args...; T, kwargs...)
end

"""
Update Mesh - necessary for composite meshes
deprecated (for now)
"""
function update_mesh!(tnm::TanMesh)
  lingrid = UnitRange(0, tnm.L - 1)
  tnm.mesh = tangrid.(lingrid, tnm.a, tnm.b, tnm.s) .+ tnm.xdens
  tnm.op = NonlinearMeshOps(tnm.mesh)
end

"""
    invert_mesh(tnm::TanMesh{T}, y::Real) where {T}

Get index of y on the mesh (not rounded - i.e. allow for interpolated values)

See also [`TanMesh`](@ref).
"""
function invert_mesh(tnm::TanMesh{T}, y::Real) where {T}
  y = convert(T, y)
  # add 1 to convert from 0:L-1 unit range to 1:L index
  return (atan((y - tnm.xdens)/tnm.s) - tnm.b)/tnm.a + one(T)
end
