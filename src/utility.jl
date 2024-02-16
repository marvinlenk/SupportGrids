"""
    _zeropad!(padded::AbstractVector, u::AbstractVector; fillzero::Bool=false)

Copies `u` to the front of `padded`.
Fills the rest of `padded` with zeros if `fillzero` is `true.

This is inspired by the `_zeropad!` function of DSP`, which we do not import here to make
the package more lightweight. This version is simpler but suffices for `SupportGrids`.

Note: this uses `unsafe_copyto!`.
"""
@inline function _zeropad!(padded::AbstractVector, u::AbstractVector; fillzero::Bool=false)
  u_len = length(u)
  @views padded_data, padded_zeros = padded[1:u_len], padded[u_len+1:end]
  if fillzero fill!(padded_zeros, 0) end
  copyto!(padded_data, u)
  
  return padded
end
