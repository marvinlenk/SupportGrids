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