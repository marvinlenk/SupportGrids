"""
Non-linear mesh, logarithmic from x0-c to x0+c, tangens towards the boundary
"""
mutable struct InvLogTanMesh{T<:Real} <: AbstractNonlinearMesh{T}
  xmin::T       # Left bound
  xmax::T       # Right bound
  x0::T         # center - not necessary a point of the grid
  σ::T          # width of the grid
  L::Int        # Number of points
  logfrac::Real # fraction of points in the log mesh
  L2::Int       # half the Number of points
  Llog::Int     # Points in the logarithmic grid
  Ltan::Int     # Points in the tangens grid
  Ltan2::Int    # Points in the tangens grid / 2
  xover::T      # crossover from log to tan
  xdens::T      # dense point of the tan grid

  invlogmesh::InvLogMesh{T}
  tanmesh::TanMesh{T}

  mesh::Vector{T}
  op::NonlinearMeshOps{T}

  function InvLogTanMesh(x0::Real, σ::Real, xover::Real, xdens::Real, L::Int;
      tan_s::Real=0.5, log_s::Real=0.001, logfrac::Real=2, sclose::Real=1e-9,
      T::Type{<:Real}=Float64)
    Llog = round(Int, L/logfrac)
    Llog += Llog % 2 == 0 ? 0 : 1
    Ltan = Int(L - Llog)
    Ltan2 = Int(Ltan/2)
    xmin = x0 - σ/2.0
    xmax = x0 + σ/2.0
  
    invlogmesh = InvLogMesh(x0, 2*xover, Llog, s=log_s, sclose=sclose)
    t0 = invlogmesh[end] + (invlogmesh[end] - invlogmesh[end-1])/tan_s
    t0 = t0 < (xdens + xover)/2 ? t0 : (xover + xdens)/2
    tanmesh = TanMesh(t0, xdens, σ/2.0, Ltan2, s=tan_s)
  
    mesh = [-reverse(tanmesh.mesh); invlogmesh.mesh; tanmesh.mesh] .+ x0
    op = NonlinearMeshOps(mesh)
    new{T}(xmin, xmax, x0, σ, L, logfrac, L/2, Llog, Ltan, Ltan2, xover, xdens, invlogmesh, tanmesh, mesh, op)
  end
  #
  InvLogTanMesh(::Type{T}, args...; kwargs...) where {T<:Real} = InvLogTanMesh(args...; kwargs..., T=T)
end

"""
Update Mesh - necessary for composite meshes
"""
function update_mesh!(ltm::InvLogTanMesh)
  @unpack tanmesh, invlogmesh, x0, σ, L, logfrac, Llog, Ltan = ltm
  ltm.Llog = round(Int, L/logfrac)
  ltm.Llog += ltm.Llog % 2 == 0 ? 0 : 1
  ltm.Ltan = Int(L - Llog)
  ltm.Ltan2 = Int(Ltan/2)
  ltm.xmin = x0 - σ/2.0
  ltm.xmax = x0 + σ/2.0
  update_mesh!(tanmesh)
  update_mesh!(invlogmesh)
  
  ltm.mesh = [-reverse(tanmesh.mesh); invlogmesh.mesh; tanmesh.mesh] .+ x0
  ltm.op = NonlinearMeshOps(ltm.mesh)
end

"""
Get index of y on the mesh (not rounded)
"""
function invert_mesh(ltm::InvLogTanMesh{T}, y::Real) where {T}
  y = convert(T, y)
  # Loggrid goes from -xover to xover, everything else is Tanggrid 
  # tangens grid is defined positive and then mirrored to negative
  y < -ltm.xover && return ltm.Ltan2 - invert_mesh(ltm.tanmesh, -y) + 1
  y > ltm.xover && return invert_mesh(ltm.tanmesh, y) + ltm.Ltan2 + ltm.Llog
  return invert_mesh(ltm.invlogmesh, y) + ltm.Ltan2
end
