############################################
# Multiple dispatch for functions on Grids #
############################################
Base.firstindex(grid::SupportGrid) = 1
Base.lastindex(grid::SupportGrid) = grid.len
Base.length(grid::SupportGrid) = grid.len
Base.iterate(grid::SupportGrid, state=1) = iterate(grid.points, state)
Base.getindex(grid::SupportGrid, i...) = grid.points[i...]
Base.minimum(grid::SupportGrid) = grid.xmin
Base.maximum(grid::SupportGrid) = grid.xmax
Base.range(grid::SupportGrid) = grid.points
Base.collect(grid::SupportGrid) = grid.points
Base.eltype(grid::SupportGrid) = eltype(grid.points)
Base.eachindex(grid::SupportGrid) = eachindex(grid.points)
-(grid::SupportGrid) = -grid.points
+(grid::SupportGrid) = +grid.points
Base.show(io::IO, grid::SupportGrid) = print_gridparams(io, grid)

"""
    *(grid::SupportGrid, u::AbstractArray) = integrate(grid, u)

See also [`integrate`](@ref).
"""
*(grid::SupportGrid, u::AbstractArray; kwargs...) = integrate(grid, u; kwargs...)

# Make Grids callable
(grid::SupportGrid)(x::Int) = grid.points[x]
(grid::SupportGrid)(x::UnitRange) = grid.points[x]


#######################
# Integration methods #
#######################

"""
    simpson_coefficients(i::Int, differences::AbstractVector)

Helper function to extract coefficients for Simpson's rule.
See [Wikipedia](https://en.wikipedia.org/wiki/Simpson%27s_rule) for details.

See also [`integration_weights`](@ref), ['simpson_coefficients_odd'](@ref).
"""
@inline function simpson_coefficients(i::Int, differences::AbstractVector)
  # weights are offset by one
  h₂ᵢ, h₂ᵢ₊₁ = differences[2i + 1], differences[2i + 2]
  α = ( 2 * h₂ᵢ₊₁^3 - h₂ᵢ^3 + 3h₂ᵢ * h₂ᵢ₊₁^2 ) / ( 6h₂ᵢ₊₁ * (h₂ᵢ₊₁ + h₂ᵢ) )
  β = ( h₂ᵢ₊₁^3 + h₂ᵢ^3 + 3h₂ᵢ₊₁ * h₂ᵢ * (h₂ᵢ₊₁ + h₂ᵢ) ) / ( 6h₂ᵢ₊₁ * h₂ᵢ )
  η = ( 2 * h₂ᵢ^3 - h₂ᵢ₊₁^3 + 3h₂ᵢ₊₁ * h₂ᵢ^2 ) / ( 6h₂ᵢ * (h₂ᵢ₊₁ + h₂ᵢ) )
  return α, β, η
end

"""
    simpson_coefficients_odd(i::Int, differences::AbstractVector)

Helper function to correct the boundary values for grids of odd length.
See [Wikipedia](https://en.wikipedia.org/wiki/Simpson%27s_rule) for details.

[`integration_weights`](@ref), ['simpson_coefficients'](@ref).
"""
@inline function simpson_coefficients_odd(differences::AbstractVector)
  # weights are offset by one
  hₙ₋₁, hₙ₋₂ = differences[end-1], differences[end-2]
  α = ( 2 * hₙ₋₁^2 + 3hₙ₋₁ * hₙ₋₂ ) / ( 6* (hₙ₋₂ + hₙ₋₁) )
  β = ( hₙ₋₁^2 + 3hₙ₋₁ * hₙ₋₂ ) / ( 6hₙ₋₂ )
  η = -( hₙ₋₁^3 ) / ( 6hₙ₋₂ * (hₙ₋₂ + hₙ₋₁) )
  return α, β, η
end

"""
    integration_weights(points::AbstractVector, method::Symbol=:trapz)

Calculates integration weights for an integral over `points` using a given `method`.

# Supported `method` symbols
- `:trapz` Trapezoidal rule
- `:simps` Basic Simpson's 1/3 rule

See also [`integrate`](@ref), ['simpson_coefficients'](@ref).
"""
@inline function integration_weights(points::AbstractVector, method::Symbol=:trapz)
  @assert method in (:trapz, :simps) "Expected method to be :trapz or :simps"
  
  if method === :trapz
    weights = [
        points[2] - points[1];
        points[3:end] .- points[1:end-2];
        points[end] - points[end-1]
      ] ./ 2
  elseif method === :simps
    L = length(points)
    differences = points[2:end] .- points[1:end-1]
    coefficients = zeros(L÷2 - 1, 3)
    for i in UnitRange(0, L÷2 - 2)
      coefficients[i+1, :] .= simpson_coefficients(i, diffs)
    end
    weights = zeros(L)
    @views weights[1:2:end-3] += coefficients[:, 3]
    @views weights[2:2:end-2] += coefficients[:, 2]
    @views weights[3:2:end-1] += coefficients[:, 1]
    
    # For odd length, add correction
    if L % 2 == 0
      weights[end-2:end] .+= reverse(simpson_coefficients_odd(diffs))
    end
  end
  
  if any(isinf.(weights))
    @warn "Integration weights contain Infs, duplicate entries in the grid?"
  end
  
  return weights
end

# concider using FiniteDifferences in the future
# Wrapper for SupportGrid input
integration_weights(grid::SupportGrid, args...; kwargs...
  ) = integration_weights(grid.points, args...; kwargs...)

"""
    integrate!(out::AbstractArray, grid::SupportGrid, u::AbstractArray; dims::Int=1)
Writes the integral of `u` over `grid` to`out` using dimension permutation, reshaping and
matrix multiplication. Integration over first dimension is most efficient.

See also [`integrate`](@ref), [`LinearAlgebra.dot`](@ref).
"""
function integrate!(out::AbstractArray, grid::SupportGrid, u::AbstractArray; dims::Int=1)
  @assert size(x) == size(u)
  @assert dims <= ndims(u)
  weights = grid.op.weights
  n = prod(size(u)[1:end .!= dims]) # leftover dimensions flattened size
  
  # permute such that integrated dimension is in first place
  u_permuted = if dims==1
    u
  else
    permutedims(u, [dims, collect(1:ndims(u))[1:end .!= dims]...])
  end
  
  # flatten non-integrated dimensions
  u_reshaped = reshape(u_permuted, size(u, dims), n)
  out_reshaped = reshape(out, n)
  
  # take the integrals
  out_reshaped .= [@views weights ⋅ u_reshaped[:,i] for i=1:n]
  
  return out
end

"""
    integrate(grid::SupportGrid, u::AbstractArray; dims::Int=1)
Returns the integral of `u` over `grid` via dimension permutation, reshaping and matrix
multiplication. Integration over first dimension is most efficient.

See also [`integrate!`](@ref), [`LinearAlgebra.dot`](@ref).
"""
integrate(grid::SupportGrid, u::AbstractArray; dims::Int=1
  ) = integrate!(zeros(eltype(u), size(u)[1:end .!= dims]), grid, u; dims)

"""
    integrate(grid::SupportGrid, u::AbstractVector)
Writes the integral of `u` over `grid` to `out` using the `LinearAlgebra` dot-product.

See also [`LinearAlgebra.dot`](@ref).
"""
integrate(grid::SupportGrid, u::AbstractVector) = grid.op.weights ⋅ u


##################
# Printing rules #
##################
"""
Printing rules
"""
function print_gridparams(io::IO, grid::SupportGrid)
  for i in 1:nfields(grid)
    field = getfield(grid, i)
    fname = (fieldname(typeof(grid), i))
    ft = typeof(field)
    if ft <: SupportGrid
      println(io, "--- $fname ---")
      printGridparams(io, getfield(Grid, i))
    elseif !(fname == :points || fname == :op)
      field = getfield(grid, i)
      if ft <: Tuple && eltype(field) <: SupportGrid
        for el in field
          println(io, "--- $(typeof(el)) ---")
          print_gridparams(io, el)
        end
      else
        println(io, "$fname = $(field)")
      end
    end
  end
end

print_gridparams(grid::SupportGrid) = print_gridparams(Base.stdout, grid)
