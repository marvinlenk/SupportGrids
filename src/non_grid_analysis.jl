# Needs some love
function derivative(grid::SupportGrid, u::AbstractVector)
  return integration_weights(u, :trapz) ./ integration_weights(grid, :trapz)
end

"""
    integrate(u::AbstractVector, v::AbstractArray; integration_type=:trapz)

Numerical integration of `v` over the support `u` with a given `integration_type` supported by [`integration_weights`](@ref).

See also [`integration_weights`](@ref).
"""
function integrate(u::AbstractVector, v::AbstractArray; integration_type=:trapz)
  return if ndims(v) == 1
    sum(integration_weights(u, integration_type) .* v)[1]
  else
    reshape( sum(integration_weights(u, integration_type) .* v), size(v)[2:end])
  end
end
