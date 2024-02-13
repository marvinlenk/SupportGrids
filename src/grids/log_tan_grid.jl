"""
Non-linear mesh, logarithmic from x0-c to x0+c, tangens towards the boundary
"""
mutable struct LogTanMesh{T<:Real} <: AbstractNonlinearMesh{T}
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

  logmesh::LogMesh{T}
  tanmesh::TanMesh{T}

  mesh::Vector{T}
  op::NonlinearMeshOps{T}

  function LogTanMesh(x0::Real, σ::Real, xover::Real, xdens::Real, L::Int;
      tan_s::Real=0.7, log_s::Real=0.1, logfrac::Real=2, sclose::Real=1e-9,
      T::Type{<:Real}=Float64)
    Llog = round(Int, L/logfrac)
    Llog += Llog % 2 == 0 ? 0 : 1
    Ltan = Int(L - Llog)
    Ltan2 = Int(Ltan/2)
    xmin = x0 - σ/2.0
    xmax = x0 + σ/2.0
    # create generic tan mesh and change scaling parameters  
#     tanmesh = TanMesh(xover, xdens, σ/2.0, Ltan2, tan_s)
#     ymin_eff = (xover - xdens)/tanmesh.s
#     ymax_eff = (xmax - xdens)/tanmesh.s
#     tanmesh.a = (atan(ymax_eff) - atan(ymin_eff) ) / Ltan2
#     tanmesh.b = atan(ymin_eff) + tanmesh.a
    
#     # create generic log mesh and change scaling parameters
#     # choose the logmesh s so that derivatives are smooth
#     tanslope = tanmesh.a * tanmesh.s / (cos(tanmesh.b - tanmesh.a))^2
#     logmesh = LogMesh(zero(T), 2*xover, Llog, tan_s)
#     function s_root(s)
#       c = tanslope
#       N = Llog/2.0
#       a = (3.0/2.0 - N - s / c)
#       a /= (s / c + N - 1.0)^2
#       b = -a
#       b *= s / c + N - 1.0
#       return exp(-(xover)/s) - (a * (N - 1) + b)
#     end
#     logmesh.s = find_zero(s_root, ((xover - x0)/10, (xover - x0)*10), Bisection())
#     logmesh.a = (3.0/2.0 - Llog/2.0 - logmesh.s / tanslope)
#     logmesh.a /= (logmesh.s / tanslope + Llog/2.0 - 1.0)^2
#     logmesh.b = -logmesh.a
#     logmesh.b *= logmesh.s / tanslope + Llog/2.0 - 1.0
    
#     update_mesh!(tanmesh)
#     update_mesh!(logmesh)
  
    logmesh = LogMesh(x0, 2*xover, Llog, s=log_s, sclose=sclose)
    t0 = 2logmesh[end] - logmesh[end-1] 
    t0 = t0 < (xdens + xover)/2 ? t0 : (xover + xdens)/2
    tanmesh = TanMesh(t0, xdens, σ/2.0, Ltan2, s=tan_s)
  
    mesh = [-reverse(tanmesh.mesh); logmesh.mesh; tanmesh.mesh] .+ x0
    op = NonlinearMeshOps(mesh)
    new{T}(xmin, xmax, x0, σ, L, logfrac, L/2, Llog, Ltan, Ltan2, xover, xdens, logmesh,
      tanmesh, mesh, op)
  end
  #
  LogTanMesh(::Type{T}, args...; kwargs...) where {T<:Real} = LogTanMesh(args...; kwargs..., T=T)
end

"""
Update Mesh - necessary for composite meshes
"""
function update_mesh!(ltm::LogTanMesh)
  @unpack tanmesh, logmesh, x0, σ, L, logfrac, Llog, Ltan = ltm
  ltm.Llog = round(Int, L/logfrac)
  ltm.Llog += ltm.Llog % 2 == 0 ? 0 : 1
  ltm.Ltan = Int(L - Llog)
  ltm.Ltan2 = Int(Ltan/2)
  ltm.xmin = x0 - σ/2.0
  ltm.xmax = x0 + σ/2.0
  update_mesh!(tanmesh)
  update_mesh!(logmesh)
  
  ltm.mesh = [-reverse(tanmesh.mesh); logmesh.mesh; tanmesh.mesh] .+ x0
  ltm.op = NonlinearMeshOps(ltm.mesh)
end

"""
Get index of y on the mesh (not rounded)
"""
function invert_mesh(ltm::LogTanMesh{T}, y::Number) where {T}
  y = convert(T, y)
  # Loggrid goes from -xover to xover, everything else is Tanggrid 
  # tangens grid is defined positive and then mirrored to negative
  y < -ltm.xover && return ltm.Ltan2 - invert_mesh(ltm.tanmesh, -y) + 1
  y > ltm.xover && return invert_mesh(ltm.tanmesh, y) + ltm.Ltan2 + ltm.Llog
  return invert_mesh(ltm.logmesh, y) + ltm.Ltan2
end
