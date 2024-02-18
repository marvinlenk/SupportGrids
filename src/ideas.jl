# Apparently only faster for very large arrays, not relevant
"""
    integrate(grid::SupportGrid, u::AbstractArray; dims::Int=1)
Returns the integral of `u` over `grid` along the dimension `dims` via matrix multiplication.

See also [`LinearAlgebra.dot`](@ref).
"""
function integrate(grid::SupportGrid, u::AbstractArray; dims::Int=1)
  @assert dims <= ndims(u)
  weights = grid.op.weights
  u_size = size(u)
  dims_mask = [true for i=1:ndims(u)]
  dims_mask[dims] = false
  
  u_size_masked = u_size[dims_mask]
  n = prod(u_size_masked)
  
  # permute such that integrated dimension is in first place
  u_permuted = if dims==1
    u
  else
    permutedims(u, [dims, collect(1:ndims(u))[dims_mask]...])
  end
  
  # flatten non-integrated dimensions
  u_reshaped = reshape(u_permuted, (u_size[dims], n))
  
  # integrate all flattened dimensions and unflatten them
  # note that this method makes it not efficient to inline this function
  return reshape([weights ⋅ u_reshaped[:,i] for i=1:n], u_size_masked)
end

# So this is better but not quite there
"""
    integrate(grid::SupportGrid, u::AbstractArray; dims::Int=1)
Returns the integral of `u` over `grid` along the dimension `dims`.
"""
function integrate(grid::SupportGrid, u::AbstractArray; dims::Int=1)
  @assert dims <= ndims(u)
  u_size = size(u)
  dims_mask = [true for i=1:ndims(u)]
  dims_mask[dims] = false
  u_size_masked = u_size[dims_mask]
  
  # permute such that integrated dimension is in first place
  u_permuted = if dims==1
    u
  else
    permutedims(u, [dims, collect(1:ndims(u))[dims_mask]...])
  end
  
  return reshape(sum(w .* u_permuted, dims=1), u_size_masked)
end

# Not sure if this should be in here
"""
Kramers-Kronig transformation of a function defined on a linear grid.

NOTE: `u` is assumed to be analytic in the lower half-plane. Multiply the 
`:imag` part by (-1) for functions analytic in the upper half-plane.
"""
function kramerskronig(lingrid::LinearGrid, u::AbstractVector{<:Real}, u_out::Symbol)
  @assert u_out === :real || u_out === :imag
  return hilbert(lingrid, ifelse(u_out === :real, u, -u))
end
function kramerskronig!(lingrid::LinearGrid, u::AbstractVector{Complex{T}}, u_out::Symbol) where {T<:Real}
  u = reinterpret(T, u)
  @views u1, u2 = u[1:2:end], u[2:2:end]

  if u_out === :real
    u1 .= kramerskronig(lingrid, u2, u_out)
  else
    u2 .= kramerskronig(lingrid, u1, u_out)
  end
end




# not sure if we want this
"""
Scaling function for the logarithmic grid - x0 has to be added
"""
_lineargrid(i, a, b) = a * i + b

"""
Scaling function for the logarithmic grid - x0 has to be added
"""
function _lineargrid_params(xmin, xmax, len)
  b = xmin
  a = (xmax - b) / (len - 1)
  return a, b
end

