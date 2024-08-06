######################
# Definite integrals #
######################

"""
    integral!(grid::SupportGrid, out::AbstractArray, u::AbstractArray; dims::Int=1)
Writes the integral of `u` over `grid` to `out` using reshaping and weighted sums.
Integration over first dimension is most efficient.

See also [`integral`](@ref), [`antiderivative!`](@ref), [`reshape`](@ref), [`sum`](@ref).
"""
function integral!(grid::SupportGrid, out::AbstractArray, u::AbstractArray; dims::Int=1)
  @assert size(out) == size(u)[1:end .!= dims]
  @assert dims <= ndims(u)
  weights = grid.op.weights
  # leftover dimensions flattened size
  n_before = dims == 1 ? 1 : prod(size(u)[1:dims-1])
  n_after = dims == ndims(u) ? 1 : prod(size(u)[dims+1:end])
  
  # flatten non-integrated dimensions
  u_reshaped = reshape(u, n_before, size(u, dims), n_after)
  out_reshaped = reshape(out, n_before, n_after)
  
  # take the integrals
  for j=1:n_after, i=1:n_before
    @views u_v = u_reshaped[i,:,j]
    sumfunc = j -> weights[j] * u_v[j]
    out_reshaped[i,j] = sum(sumfunc, eachindex(u_v))
  end
  
  return out
end

"""
    integral(grid::SupportGrid, u::AbstractArray; dims::Int=1)
Returns the integral of `u` over `grid` using reshaping and weighted sums.
Integration over first dimension is most efficient.

See also [`integral!`](@ref), [`antiderivative`](@ref), [`reshape`](@ref), [`sum`](@ref).
"""
integral(grid::SupportGrid, u::AbstractArray{T}; dims::Int=1
  ) where T = integral!(grid, Array{T}(undef, size(u)[1:end .!= dims]), u; dims)

"""
    integral(grid::SupportGrid, u::AbstractVector)
Returns the integral of `u` over `grid` using the `LinearAlgebra` dot-product.

See also [`LinearAlgebra.dot`](@ref), [`antiderivative`](@ref).
"""
integral(grid::SupportGrid, u::AbstractVector) = grid.op.weights ⋅ u

########################
# Indefinite integrals #
########################

"""
    antiderivative!(grid::SupportGrid, out::AbstractArray, u::AbstractArray;
                    dims::Int=1, c=0.)
Writes the antiderivative of `u` over `grid` to `out` using reshaping and a generalized
inplace cumulative sum (`accumulate!`), where `c` is the integration constant.
Antiderivative over first dimension is most efficient.

See also [`antiderivative`](@ref), [`integral!`](@ref), [`derivative!`](@ref),
[`accumulate!`](@ref).
"""
function antiderivative!(grid::SupportGrid, out::AbstractArray, u::AbstractArray;
    dims::Int=1, c=0.)
  @assert size(out) == size(u)
  @assert dims <= ndims(u)
  weights = grid.op.weights
  # leftover dimensions flattened size
  n_before = dims == 1 ? 1 : prod(size(u)[1:dims-1])
  n_after = dims == ndims(u) ? 1 : prod(size(u)[dims+1:end])
  
  # flatten non-integrated dimensions
  u_reshaped = reshape(u, n_before, size(u, dims), n_after)
  out_reshaped = reshape(out, n_before, size(out, dims), n_after)
  
  # take the integrals
  for j=1:n_after, i=1:n_before
    @views u_v, out_v = u_reshaped[i,:,j], out_reshaped[i,:,j]
    accfunc = (x,j) -> x + weights[j] * u_v[j]
    accumulate!(accfunc, out_v, eachindex(u_v), init=weights[1] * u_v[1] + c)
  end
  
  return out
end

"""
    antiderivative!(grid::SupportGrid, out::AbstractVector, u::AbstractVector; c=0.)
Writes the antiderivative of `u` over `grid` to `out` using a generalized inplace cumulative
sum (`accumulate!`), where `c` is the integration constant.

See also [`antiderivative`](@ref), [`integral!`](@ref), [`derivative!`](@ref),
[`accumulate!`](@ref).
"""
function antiderivative!(grid::SupportGrid, out::AbstractVector, u::AbstractVector; c=0.)
  @assert size(out) == size(u)
  weights = grid.op.weights
  accfunc = (x,j) -> x + weights[j] * u[j]
  accumulate!(accfunc, out, eachindex(u), init=weights[1] * u[1] + c)
  
  return out
end

"""
    antiderivative(grid::SupportGrid, u::AbstractArray; dims::Int=1, c=0.)
Returns the antiderivative of `u` over `grid` to `out` using reshaping and a generalized
inplace cumulative sum (`accumulate!`), where `c` is the integration constant.
Antiderivative over first dimension is most efficient.

See also [`antiderivative!`](@ref), [`integral`](@ref), [`derivative`](@ref),
[`accumulate`](@ref).
"""
antiderivative(grid::SupportGrid, u::AbstractArray; dims::Int=1, c=0.
  ) = antiderivative!(grid, similar(u), u; dims, c)

"""
    antiderivative(grid::SupportGrid, u::AbstractVector; c=0.)
Returns the antiderivative of `u` over `grid` to `out` using a generalized cumulative sum
(`accumulate`), where `c` is the integration constant.

See also [`antiderivative!`](@ref), [`integral`](@ref), [`derivative`](@ref),
[`accumulate`](@ref).
"""
function antiderivative(grid::SupportGrid, u::AbstractVector; c=0.)
  weights = grid.op.weights
  accfunc = (x,j) -> x + weights[j] * u[j]
  return accumulate(accfunc, out, eachindex(u), init=weights[1] * u[1] + c)
end

################################
# Lazy Operations on functions #
################################
integral(grid::SupportGrid, u::Function; kwargs...) = integral(grid, u.(grid); kwargs...)

integral!(grid::SupportGrid, out, u::Function; kwargs...
  ) = integral!(grid, out, u.(grid); kwargs...)

convolution(grid::SupportGrid, u::Function, v
  ) = convolution(grid, out, u.(grid), v)

convolution!(grid::SupportGrid, out, u, v::Function
  ) = convolution!(grid, out, u, v.(grid))

crosscorrelation(grid::SupportGrid, u::Function, v
  ) = crosscorrelation(grid, u.(grid), v)

crosscorrelation!(grid::SupportGrid, out, u, v::Function
  ) = crosscorrelation!(grid, out, u, v.(grid))

hilbert(grid::SupportGrid, u::Function,
  ) = hilbert(grid, u.(grid), v)

hilbert!(grid::SupportGrid, out, u::Function,
  ) = hilbert!(grid, out, u.(grid), v)
