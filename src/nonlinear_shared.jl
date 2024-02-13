"""
Data structure for operations on nonlinear meshes
"""
mutable struct NonlinearMeshOps{T<:AbstractFloat}
  weights::Vector{T}          # weights for integration
  convmat_freq::Array{T, 2}   # frequencies for conv (ω - ϵ), transpose of (ϵ - ω)
  xcorrmat_freq::Array{T, 2}  # frequencies for xcorr (ω + ϵ)
  kkkernel::Array{T, 2}       # Kramers-Kronig Kernel
  kkbound::Vector{T}          # Kramers-Kronig boundary term

  function NonlinearMeshOps(mesh::AbstractVector{T}; method=:trapz) where {T<:AbstractFloat}
    weights = integration_weights(mesh, method)
    
    kkkernel = [ω - ϵ == 0 ? zero(T) : weights[ϵ]/(mesh[ω] - mesh[ϵ])
      for ω in eachindex(mesh), ϵ in eachindex(mesh)] ./ π
#     kkkernel = [ω - ϵ == zero(T) ? zero(T) : one(T)/(ω - ϵ) for ω in mesh, ϵ in mesh] .* transpose(weights) ./ pi
#     kkkernel = [ω - ϵ == 0 ? zero(T) : (ϵ < ω ? (mesh[ϵ+1] - mesh[ϵ]) : (mesh[ϵ] - mesh[ϵ-1]))/(mesh[ω] - mesh[ϵ])
#       for ω in eachindex(mesh), ϵ in eachindex(mesh)] ./ pi
    kkbound = log.(abs.( (maximum(mesh) .- mesh)./(minimum(mesh) .- mesh) )) ./ pi
    kkbound[1] = kkbound[2] + (kkbound[2] - kkbound[3]) * (mesh[1] - mesh[2]) / (mesh[2] - mesh[3])
    kkbound[end] = kkbound[end-1] + 
        (kkbound[end-1] - kkbound[end-2]) * (mesh[end] - mesh[end-1]) / (mesh[end-1] - mesh[end-2])
    convmat_freq = mesh .- transpose(mesh)
    xcorrmat_freq = mesh .+ transpose(mesh)
    
    new{T}(weights, convmat_freq, xcorrmat_freq, kkkernel, kkbound)
  end
end

# overwrite numerical derivative cause it's easier here
# note that this doesnt work for simpsons method coefficients
function numerical_derivative(mesh::AbstractNonlinearMesh, u::Vector{<:Number})
  return integration_weights(u, :trapz) ./ mesh.op.weights
end

# shell
mutable struct NonlinearMeshShell{T<:AbstractFloat} <: AbstractNonlinearMesh{T}
  mesh::Vector{T}          # weights for integration
  op::NonlinearMeshOps{T}
  L::Int
  xmin::T
  xmax::T
  invert_extrapolator::AbstractExtrapolation
  
  function NonlinearMeshShell(mesh::AbstractVector{T}) where {T<:AbstractFloat}
    L = length(mesh)
    @assert issorted(mesh) || all(sortperm(mesh) .== L:-1:1) "Mesh needs to be sorted"
    
    invert_extrapolator = if issorted(mesh)
      linear_interpolation(mesh, 1:L, extrapolation_bc=Line()) 
    else
      linear_interpolation(reverse(mesh), L:-1:1, extrapolation_bc=Line())
    end
    xmin = minimum(mesh)
    xmax = maximum(mesh)
    
    new{T}(mesh, NonlinearMeshOps(mesh), L, xmin, xmax, invert_extrapolator)
  end
end

invert_mesh(nlm::NonlinearMeshShell, y::Number) = nlm.invert_extrapolator(y)


"""
Create Matrix, which rows are a function evaluated at ω - ϵ.
Each ω is the outer frequency on the same grid.
"""
function convmat(nlm::AbstractNonlinearMesh, f::Function)
  @unpack L = nlm
  T = typeof( f(nlm[1]) )
  res = zeros(T,L,L)
  convmat!(nlm, res, f)
  return res
end

@inline function convmat!(nlm::AbstractNonlinearMesh, dest::AbstractArray{T, 2},
        f::Function) where {T}
  @unpack convmat_freq, weights = nlm.op
  dest .= broadcast(f, convmat_freq) .* transpose(weights)
end


"""
Create Matrix, whose rows are a function evaluated at ω + ϵ.
Each ω is the outer frequency on the same grid.
"""
function xcorrmat(nlm::AbstractNonlinearMesh, f::Function)
  @unpack L = nlm
  T = typeof( f(nlm[1]) )
  res = zeros(T,L,L)
  xcorrmat!(nlm, res, f)
  return res
end


@inline function xcorrmat!(nlm::AbstractNonlinearMesh, dest::AbstractArray{T, 2},
        f::Function) where {T}
  @unpack xcorrmat_freq, weights = nlm.op
  dest .= broadcast(f, xcorrmat_freq) .* transpose(weights)
end

"""
Create Matrix, whose rows are a function evaluated at ϵ - ω.
Each ω is the outer frequency on the same grid.
"""
function xcorrmat_neg(nlm::AbstractNonlinearMesh, f::Function)
  @unpack L = nlm
  T = typeof(f(nlm[1]))
  res = zeros(T,L,L)
  xcorrmat_neg!(nlm, res, f)
  return res
end

@inline function xcorrmat_neg!(nlm::AbstractNonlinearMesh, dest::AbstractArray{T, 2},
        f::Function) where {T}
  @unpack convmat_freq, weights = nlm.op
  dest .= broadcast(f, transpose(convmat_freq)) .* transpose(weights)
end

"""
Contains the necessary arrays for convolution and cross-correlation calculation
"""
mutable struct NonlinearMeshMatrix{nlm<:AbstractMesh, T<:Number}
  array::Vector{T}
  convmat::Array{T, 2}  # Matrix that contains the ω - ϵ vectors as rows
  xcorrmat::Array{T, 2}   # Matrix that contains the ω + ϵ vectors as rows
  xcorrmat_neg::Array{T, 2} # Matrix that contains the ϵ - ω vectors as rows
  
  function NonlinearMeshMatrix(nlm::M, f::Function) where {M<:AbstractNonlinearMesh}
    @unpack mesh, op = nlm
    @unpack convmat_freq, xcorrmat_freq, weights = op
    T = typeof( f(mesh[1]) )
    t1 = Threads.@spawn map(f, mesh)
    t2 = Threads.@spawn convmat(nlm, f)
    t3 = Threads.@spawn xcorrmat(nlm, f)
    t4 = Threads.@spawn xcorrmat_neg(nlm, f)
    wait.([t1, t2, t3, t4])
    array = fetch(t1)
    convm = fetch(t2)
    xcorrm = fetch(t3)
    xcorrm_neg = fetch(t4)
    new{M, T}(array, convm, xcorrm, xcorrm_neg)
  end
end

# replace the values of those generated with the function f
function NonlinearMeshMatrix!(nlm::AbstractMesh, meshmat::NonlinearMeshMatrix{<:AbstractMesh, T},
    f::Function) where {T<:AbstractFloat}
  @unpack mesh, op = nlm
  @unpack convmat_freq, xcorrmat_freq, weights = op
  @unpack array, convmat, xcorrmat, xcorrmat_neg = meshmat
  # check if new values are type compatible or can be converted
  oftype(one(T), f(mesh[1]))
  # replace old values by new ones
  t1 = Threads.@spawn map!(x -> f(x), array, mesh)
  t2 = Threads.@spawn map!(x -> f(x) , convmat, convmat_freq)
  t3 = Threads.@spawn map!(x -> f(x), xcorrmat, xcorrmat_freq)
  t4 = Threads.@spawn map!(x -> f(x ), xcorrmat_neg, transpose(convmat_freq))
  wait.([t1, t2, t3, t4])
  meshmat.xcorrmat .*= transpose(weights)
  meshmat.convmat .*= transpose(weights)
  meshmat.xcorrmat_neg .*= transpose(weights)
end

"""
Array methods of NonlinearMeshMatrix
"""
Base.firstindex(meshmat::NonlinearMeshMatrix) = firstindex(meshmat.array)
Base.lastindex(meshmat::NonlinearMeshMatrix) = lastindex(meshmat.array)
Base.length(meshmat::NonlinearMeshMatrix) = length(meshmat.array)
Base.iterate(meshmat::NonlinearMeshMatrix, state=1) = iterate(meshmat.array, state)
Base.getindex(meshmat::NonlinearMeshMatrix, i...) = meshmat.array[i...]
Base.minimum(meshmat::NonlinearMeshMatrix) = minimum(meshmat.array)
Base.maximum(meshmat::NonlinearMeshMatrix) = maximum(meshmat.array)
Base.range(meshmat::NonlinearMeshMatrix) = range(meshmat.array)
Base.collect(meshmat::NonlinearMeshMatrix) = meshmat.array


"""
Integral over a vector, weighted with the appropriate factors
"""
function integrate(nlm::AbstractNonlinearMesh, v::AbstractVector)
  @unpack weights = nlm.op
  return transpose(weights) * v
end


function integrate(nlm::M, meshmat::NonlinearMeshMatrix{M}) where {M<:AbstractNonlinearMesh}
  @unpack weights = nlm.op
  @unpack array = meshmat
  return transpose(weights) * array
end

"""
Convolution of 2 functions defined on the logarithmic mesh.
v is the function of the integrated variable,
m is a matrix, where each row is the function at the outer frequency - grid
"""
function conv(nlm::AbstractNonlinearMesh, v::AbstractVector{T}, m::AbstractArray{T,2};
  applyweights=true) where {T}
  @unpack weights = nlm.op
  return applyweights ? m * (weights .* v) : m * v
end

"""
Convolution of 2 functions defined on the logarithmic mesh.
v is the function of the integrated variable, mehmat contains the shifted vectors in a matrix form
"""
function conv(meshmat::NonlinearMeshMatrix, v::AbstractVector)
  @unpack convmat = meshmat
  return convmat * v
end

conv(v::AbstractVector, meshmat::NonlinearMeshMatrix) = conv(meshmat, v)

conv(nlm::M, meshmat::NonlinearMeshMatrix{M}, v::AbstractVector
  )  where {M <: AbstractNonlinearMesh} = conv(meshmat, v)

"""
Cross-correlation of 2 functions defined on the logarithmic mesh
v is the function of the integrated variable,
m is a matrix, where each row is the function at the outer frequency + grid
"""
xcorr(nlm::AbstractNonlinearMesh, v::AbstractVector, m::AbstractArray{T,2}; applyweights=false
  ) where {T} = conv(nlm, v, m, applyweights = applyweights)

"""
Cross-correlation of 2 functions defined on the logarithmic mesh.
v is the function of the integrated variable, meshmat contains the shifted vectors in a matrix form
"""
function xcorr(meshmat::NonlinearMeshMatrix, v::AbstractVector)
  return meshmat.xcorrmat_neg * v
end

function xcorr(v::AbstractVector, meshmat::NonlinearMeshMatrix)
   return meshmat.xcorrmat * v
end

xcorr(nlm::AbstractNonlinearMesh, meshmat::NonlinearMeshMatrix{M}, v::AbstractVector
  )  where {M<:AbstractNonlinearMesh} = xcorr(meshmat, v)

xcorr(nlm::AbstractNonlinearMesh, v::AbstractVector, meshmat::NonlinearMeshMatrix{M}
  )  where {M<:AbstractNonlinearMesh} = xcorr(v, meshmat)

"""
Kramers-Kronig transformation on some linear mesh - this is the Kramers-Kronig
for the ADVANCED functions. When the function is of the RETARDED type 
(analytic in the upper half-plane), multiply by -1 for :imag. Since the real 
part of the Green functions should be the same.

Note that this formulation is basically in-place if used in the following way:
output .= kramerskronig(...)
"""
function kramerskronig(nlm::AbstractNonlinearMesh, input::AbstractVector{T}, out_comp::Symbol
  ) where {T<:AbstractFloat}
  @assert out_comp === :real || out_comp === :imag "expected :real or :imag"
  @unpack kkbound, kkkernel = nlm.op
  L = nlm.L
  
  return (out_comp === :real ? -one(T) : one(T)) .* 
    ( kkkernel .* (input .- transpose(input)) * ones(T, L) .+ (input .* kkbound) )
end

"""
In-place Kramers-Kronig on complex array
"""
function kramerskronig!(nlm::AbstractNonlinearMesh, u::AbstractVector{Complex{T}}, out_comp::Symbol
    ) where {T<:AbstractFloat}
  @assert out_comp === :real || out_comp === :imag "expected :real or :imag"
  @unpack kkbound, kkkernel = nlm.op
  L = nlm.L
  # inplace kramerskronig replaces the part that is calculated
  inp = out_comp === :real ? imag(u) : real(u)
  @views output = out_comp === :real ? reinterpret(T, u)[1:2:end] : reinterpret(T, u)[2:2:end]
  
  output .= (out_comp === :real ? -one(T) : one(T)) .*
    ( kkkernel .* (inp .- transpose(inp)) * ones(T, L) .+ (inp .* kkbound) )
end

kramerskronig(nlm::M, um::NonlinearMeshMatrix{M}, out_comp::Symbol
  ) where {M<:AbstractMesh} = kramerskronig(nlm, um.array, out_comp)
