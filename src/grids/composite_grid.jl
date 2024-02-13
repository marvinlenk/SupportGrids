"""
    CompositeMesh(xmin::Real, xmax::Real, L::Int, meshtypes::AbstractVector{Symbol}, 
      xover::AbstractVector{<:Real};
      Lfracs::AbstractVector{<:Real}=ones(Int, length(meshtypes)), integration_method=:trapz,
      meshargs::AbstractVector=[NamedTuple() for i in 1:length(meshtypes)],
      T::Type{<:Real}=Float64, kwargs...)

Generates a composite non-linear 1D mesh of length(L) and width `σ` around `x0`.
The central part is an exponential grid of width `xover` around `x0`.
The two outer lying parts are tangens grids, 
  
# Arguments
- 

"""
mutable struct CompositeMesh{T<:Real} <: AbstractNonlinearMesh{T}
  xmin::T       # Left bound
  xmax::T       # Right bound
  x0::T         # center - not necessary a point of the grid
  σ::T          # width of the grid
  L::Int        # Number of points
  Lfracs::AbstractVector{<:Real} # fraction of points in the meshes
  Larr::Vector{<:Int} # Actual collection of mesh lengths
  xover::AbstractVector{<:Real}  # crossover points
  
  nl_meshes::Tuple
  
  mesh::Vector{T}
  op::NonlinearMeshOps{T}
  #
  function CompositeMesh(xmin::Real, xmax::Real, L::Int, meshtypes::AbstractVector{Symbol}, 
      xover::AbstractVector{<:Real};
      Lfracs::AbstractVector{<:Real}=ones(Int, length(meshtypes)), integration_method=:trapz,
      meshargs::AbstractVector=[NamedTuple() for i in 1:length(meshtypes)],
      T::Type{<:Real}=Float64, kwargs...)
    @assert allequal( (length(meshtypes), length(Lfracs), length(xover)+1) )
    σ = xmax - xmin
    x0 = xmin + σ/2
    
    # generate Larr and make sure it reproduces the correct L
    Larr = round.(Int, Lfracs .* (L / sum(Lfracs)))
    Ldiff = L - sum(Larr)
    # the difference between L and sum(Larr) is distributed equaly to the end of Larr
    Larr[end-abs(Ldiff)+1:end] .+= sign(Ldiff)
    Larr_cumsum = [0; cumsum(Larr)]
    
    nlm_arr = []
    mesh = zeros(T, L)
    x_h = [xmin; xover...; xmax]
    
    for (i,el) in enumerate(meshtypes)
      meshtype = Symbol(el,:Mesh)
      
      # all meshes except the first need an additional point to be left out for no duplicates
      # the point is removed on the left side
      notfirst = i==1 ? 0 : 1
      L_i = Larr[i] + notfirst
      xmin_i, xmax_i = x_h[i:i+1]
      
      # add new mesh to nlm_arr and the points to mesh
      push!(nlm_arr, @eval $meshtype($T, $xmin_i, $xmax_i, $L_i; $meshargs[$i]...))
     
      @. mesh[Larr_cumsum[i]+1:Larr_cumsum[i+1]] = nlm_arr[end].mesh[1+notfirst:end]
    end
    op = NonlinearMeshOps(mesh, method=integration_method; kwargs...)
    new{T}(xmin, xmax, x0, σ, L, Lfracs, Larr, xover, Tuple(nlm_arr), mesh, op)
  end
  #
  CompositeMesh(::Type{T}, args...; kwargs...) where {T<:Real} = CompositeMesh(args...; T, kwargs...)
end

function invert_mesh(cmp::CompositeMesh{T}, y::Real) where {T}
  @unpack xover, nl_meshes, Larr = cmp
  
  mesh_id = sum(y .> xover) + 1 # determines in whichs section of the mesh y is
  return sum(Larr[1:mesh_id-1]) - (mesh_id==1 ? 0 : 1) + invert_mesh(nl_meshes[mesh_id], y)
end
