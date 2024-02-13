"""
Scaling function for the inverse logarithmic grid - x0 has to be added
"""
function invloggrid(i, a, b, s)
  x = a*i + b
  return (-sign(x)/(log(abs(x)))  - 1) * s
end

"""
Weights for integration, not used so far
"""
function invlogweights(i, a, b, s)
  x = a*i + b
  return a * s * (invloggrid(i, a, b, s))^2 / (abs(x) - 1)
end

"""
Logarithmic mesh base
"""
mutable struct InvLogMesh{T<:Real} <: AbstractNonlinearMesh{T}
  xmin::T    # Left bound
  xmax::T    # Right bound
  x0::T      # center - not necessary a point of the grid
  σ::T       # width of the grid
  L::Int     # Number of points
  L2::Int    # Half the number of points
  a::T       # Scaling parameter 1
  b::T       # Scaling parameter 2
  s::T       # scaling parameter 3

  mesh::Vector{T}
  op::NonlinearMeshOps{T}

  function InvLogMesh(x0::Real, σ::Real, L::Int; s::Real=1, sclose::Real=1e-9,
      T::Type{<:Real}=Float64)
    L % 2 == 0 ? nothing : ArgumentError("Length of logarightmic mesh must be even")
    xmin = x0 - σ/2
    xmax = x0 + σ/2  
    L2 = Int(L/2)
    y0eff = (sclose / s + 1)
    ymaxeff = (σ/2 / s + 1)
    b = exp(-1/y0eff)
    a = (exp(-1/ymaxeff) - b) / (L2 - 1);
  
    lingrid = UnitRange(0, L2 - 1)
    posgrid = invloggrid.(lingrid, a, b, s)
    posgrid[end] = σ/2 
    mesh = [-reverse(posgrid); posgrid] .+ x0
    op = NonlinearMeshOps(mesh)
    new{T}(xmin, xmax, x0, σ, L, L2, a, b, s, mesh, op)
  end
  #
  InvLogMesh(::Type{T}, args...; kwargs...) where {T<:Real} = InvLogMesh(args...; kwargs..., T=T)
end
  
"""
Update Mesh - necessary for composite meshes
"""
function update_mesh!(lgm::InvLogMesh)
  lgm.L % 2 == 0 ? nothing : ArgumentError("Length of logarightmic mesh must be even")
  lgm.xmin = lgm.x0 - lgm.σ/2
  lgm.xmax = lgm.x0 + lgm.σ/2
  lgm.L2 = Int(lgm.L/2)
  
  lingrid = UnitRange(0, lgm.L2 - 1)
  posgrid = invloggrid.(lingrid, lgm.a, lgm.b, lgm.s)
  lgm.mesh = [-reverse(posgrid); posgrid] .+ lgm.x0
  lgm.op = NonlinearMeshOps(lgm.mesh)
end

"""
Get index of y on the mesh (not rounded - i.e. allow for interpolated values)
"""
function invert_mesh(lgm::InvLogMesh{T}, y::Real) where {T}
  y = convert(T, y)
  # add 1 to convert from 0:L-1 unit range to 1:L index
  # loggrid is created on positive y + y0 and mirrored
  y - lgm.x0 > zero(T) && return (exp(-1 / ((y - lgm.x0)/lgm.s + 1)) - lgm.b)/lgm.a + one(T) + T(lgm.L2)
  return T(lgm.L2) - (exp(-1 / ((-y + lgm.x0)/lgm.s + 1)) - lgm.b)/lgm.a
end
