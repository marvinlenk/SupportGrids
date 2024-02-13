"""
    ExpTanMesh(x0, σ, xover, xdensL, L; <keyword arguments>)

Generates a composite non-linear 1D mesh of length(L) and width `σ` around `x0`.
The central part is an exponential grid of width `xover` around `x0`.
The two outer lying parts are tangens grids, 
  
# Arguments
- 

"""
mutable struct ExpTanMesh{T<:Real} <: AbstractNonlinearMesh{T}
  xmin::T       # Left bound
  xmax::T       # Right bound
  x0::T         # center - not necessary a point of the grid
  σ::T          # width of the grid
  L::Int        # Number of points
  Lfracs::Tuple # fraction of points in the meshes
  L2::Int       # half the Number of points
  Lexp::Int     # Points in the exponential grid
  LtanL::Int     # Points in the left tangens grid
  LtanR::Int    # Points in the right tangens grid
  xover::T      # crossover from log to tan
  xdensL::T      # dense point of the left tan grid
  xdensR::T      # dense point of the right tan grid

  expmesh::ExpMesh{T}
  tanmeshL::TanMesh{T}
  tanmeshR::TanMesh{T}

  mesh::Vector{T}
  op::NonlinearMeshOps{T}
  #
  function ExpTanMesh(x0::Real, σ::Real, xover::Real, xdensL::Real, L::Int; xdensR::Real=xdensL, 
      tan_s::Real=0.5, tanL_s::Real=tan_s, tanR_s::Real=tan_s, Lfracs=(3,5,1), sclose::Real=1e-9,
      integration_method=:trapz, T::Type{<:Real}=Float64)
    Lfsum = sum(Lfracs)
    Lexp = round(Int, L * Lfracs[2]/Lfsum)
    Lexp += Lexp % 2 == 0 ? 0 : 1
    LtanL = round(Int, L * Lfracs[1]/Lfsum)
    LtanR = Int(L - LtanL - Lexp)
    xmin = x0 - σ/2.0
    xmax = x0 + σ/2.0
  
    expmesh = ExpMesh(x0 - xover, x0 + xover, Lexp, sclose=sclose)
    t0 = expmesh[end] + (expmesh[end] - expmesh[end-1])
    tanmeshR = TanMesh(t0, σ/2.0, LtanR, xdens=xdensR, s=tanR_s)
    tanmeshL = TanMesh(t0, σ/2.0, LtanL, xdens=-xdensL, s=tanL_s)
  
    mesh = [-reverse(tanmeshL.mesh); expmesh.mesh; tanmeshR.mesh] .+ x0
    op = NonlinearMeshOps(mesh, method=integration_method)
    new{T}(xmin, xmax, x0, σ, L, Lfracs, L/2, Lexp, LtanL, LtanR, xover, xdensL, xdensR, expmesh,
      tanmeshL, tanmeshR, mesh, op)
  end
  #
  ExpTanMesh(::Type{T}, args...; kwargs...) where {T<:Real} = ExpTanMesh(args...; T, kwargs...)
end

"""
Update Mesh - necessary for composite meshes
"""
function update_mesh!(ltm::ExpTanMesh)
  @unpack tanmesh, expmesh, x0, σ, L, expfrac, Lexp, Ltan = ltm
  ltm.Lexp = round(Int, L/expfrac)
  ltm.Lexp += ltm.Lexp % 2 == 0 ? 0 : 1
  ltm.Ltan = Int(L - Lexp)
  ltm.Ltan2 = Int(Ltan/2)
  ltm.xmin = x0 - σ/2.0
  ltm.xmax = x0 + σ/2.0
  update_mesh!(tanmesh)
  update_mesh!(expmesh)
  
  ltm.mesh = [-reverse(tanmesh.mesh); expmesh.mesh; tanmesh.mesh] .+ x0
  ltm.op = NonlinearMeshOps(ltm.mesh)
end

"""
Get index of y on the mesh (not rounded)
"""
function invert_mesh(ltm::ExpTanMesh{T}, y::Real) where {T}
  @unpack xover, LtanL, LtanR, Lexp, tanmeshL, tanmeshR, expmesh = ltm
  y = convert(T, y)
  # Loggrid goes from -xover to xover, everything else is Tanggrid 
  # tangens grid is defined positive and then mirrored to negative
  y < -xover && return LtanL - invert_mesh(tanmeshL, -y) + 1
  y > xover && return invert_mesh(tanmeshR, y) + LtanL + Lexp
  return invert_mesh(expmesh, y) + LtanL
end
