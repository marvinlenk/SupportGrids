"""
Scaling function for the logarithmic grid - x0 has to be added
"""
function loggrid(i, a, b, s)
  return -log(a * i + b) * s
end

"""
Weights for integration, not used so far
"""
function logweights(i, a, b, s)
  return -a * s / (a * i + b)
end

"""
Logarithmic mesh base
"""
mutable struct LogMesh{T<:Real} <: AbstractNonlinearMesh{T}
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

  function LogMesh(x0::Real, σ::Real, L::Int; s::Real=1, sclose::Real=1e-9,
      T::Type{<:Real}=Float64)
    L % 2 == 0 ? nothing : ArgumentError("Length of logarightmic mesh must be even")
    xmin = x0 - σ/2
    xmax = x0 + σ/2  
    L2 = Int(L/2)
#     # c, d and hb are helper functions
#     c = complex((L2 - 2)/(L2 - 1))
#     d = complex(exp(-σ/(2 * s))/(L2 - 1))
#     hb = (sqrt(3) * sqrt(27 * d^2 - 4 * c^3) + 9 * d)^(1/3)
#     b = (2/3)^(1/3) * c / hb 
#     b += hb / (18^(1/3))
#     b = real(b)
#     a = real(d - b/(L2 - 1))
    
    b = exp(-sclose/s)
    a = (exp(-σ/(2 * s)) - b)/(L2 - 1)
  
    lingrid = UnitRange(0, L2 - 1)
    posgrid = loggrid.(lingrid, a, b, s)
    posgrid[end] = σ/2 
    mesh = [-reverse(posgrid); posgrid] .+ x0
    op = NonlinearMeshOps(mesh)
    new{T}(xmin, xmax, x0, σ, L, L2, a, b, s, mesh, op)
  end
  #
  LogMesh(::Type{T}, args...; kwargs...) where {T<:Real} = LogMesh(args...; kwargs..., T=T)
end
  
"""
Update Mesh - necessary for composite meshes
"""
function update_mesh!(lgm::LogMesh)
  lgm.L % 2 == 0 ? nothing : ArgumentError("Length of logarightmic mesh must be even")
  lgm.xmin = lgm.x0 - lgm.σ/2
  lgm.xmax = lgm.x0 + lgm.σ/2
  lgm.L2 = Int(lgm.L/2)
  
  lingrid = UnitRange(0, lgm.L2 - 1)
  posgrid = loggrid.(lingrid, lgm.a, lgm.b, lgm.s)
  lgm.mesh = [-reverse(posgrid); posgrid] .+ lgm.x0
  lgm.op = NonlinearMeshOps(lgm.mesh)
end

"""
Get index of y on the mesh (not rounded - i.e. allow for interpolated values)
"""
function invert_mesh(lgm::LogMesh{T}, y::Real) where {T}
  y = convert(T, y)
  # add 1 to convert from 0:L-1 unit range to 1:L index
  # loggrid is created on positive y + y0 and mirrored
  y - lgm.x0 > zero(T) && return (exp((lgm.x0 - y)/lgm.s) - lgm.b)/lgm.a + one(T) + T(lgm.L2)
  return T(lgm.L2) - (exp((y - lgm.x0)/lgm.s) - lgm.b)/lgm.a
end
