"""
Data structure for operations in data defined on a linear Grid
"""
struct LinearGridOps{T<:Real} <: AbstractGridOps{T}
  weights::Vector{T}                      # weights for integration
  FT::Plan                                # Fourier transform
  FT⁻¹::ScaledPlan                        # inv Fourier transform
  padded::Vector{<:Union{T, Complex{T}}}  # padded vector
  f1::Vector{Complex{T}}                  # output of Fourier transform
  f2::Vector{Complex{T}}                  # output of Fourier transform
  out::Vector{<:Union{T, Complex{T}}}     # output of inv Fourier transform
  conv_shift::Vector{Complex{T}}          # frequency shift for convolutions
  corr_shift::Vector{Complex{T}}          # frequency shift for correlations
  fft_1_x::Vector{Complex{T}}             # Fourier transform of 1/x

  function LinearGridOps(points::Vector{T}, complex_op::Bool; integration_method::Symbol=:trapz) where {T<:Real}
    weights = integration_weights(points, integration_method)
    L2 = nextpow(2, 2*length(points)-1) # size of padded operations
    if T <: BigFloat L2 -= 1 end

    padded = zeros(complex_op ? Complex{T} : T, L2)
    out = similar(padded)
    FT = complex_op ? plan_fft(out) : plan_rfft(out)
    f1 = zeros(Complex{T}, FT.osz)
    f2 = zeros(Complex{T}, FT.osz)
    FT⁻¹ = complex_op ? plan_ifft(f1) : plan_irfft(f1, L2)

    freqs = complex_op ? fftfreq(L2) : rfftfreq(L2)
    dx = points[2] - points[1]
    kₐ = points[1] * 2pi * freqs / dx

    # NOTE: already normalized (dx and scale)
    conv_shift = exp.(-1im * kₐ) * dx * FT⁻¹.scale
    corr_shift = conj.(conv_shift)
    fft_1_x = -1im * sign.(freqs) * FT⁻¹.scale

    new{T}(weights,FT,FT⁻¹,padded,f1,f2,out,conv_shift,corr_shift,fft_1_x)
  end
end

"""
Linear grid
"""
struct LinearGrid{T<:Real} <: SupportGrid{T}
  xmin::T              # Left bound
  xmax::T              # Right bound
  L::Int               # Number of points
  dx::T                # Frequency discretisation
  points::Vector{T}
  op::LinearGridOps{T}

  function LinearGrid(xmin::Real, xmax::Real, L::Int; complex_op=false,
      T::Type{<:Real}=Float64)
    dx = (xmax - xmin) / (L - 1)
    points = collect(range(xmin, stop=xmax, length=L))
    new{T}(xmin, xmax, L, dx, points, LinearGridOps(points, complex_op))
  end
  #
  LinearGrid(::Type{T}, args...; kwargs...) where {T<:Real} = LinearGrid(args...; kwargs..., T=T)
end

"""
Convolution of 2 functions defined on a linear grid
"""
function convolution(lingrid::LinearGrid, u::AbstractVector, v::AbstractVector)
  @unpack padded, f1, f2, out, FT⁻¹, FT, conv_shift = lingrid.op

  unsafe_execute!(FT, _zeropad!(padded, u), f1)
  unsafe_execute!(FT, _zeropad!(padded, v), f2)
  @. f2 *= f1 * conv_shift
  unsafe_execute!(FT⁻¹.p, f2, out)
  return out[1:lingrid.L]
end

"""
Cross-Correlation of 2 functions defined on a linear grid
"""
function crosscorrelation(lingrid::LinearGrid, u::AbstractVector, v::AbstractVector)
  @unpack padded, f1, f2, out, FT⁻¹, FT, corr_shift = lingrid.op

  unsafe_execute!(FT, _zeropad!(padded, u), f1)
  unsafe_execute!(FT, _zeropad!(padded, v), f2)
  @. f2 *= conj(f1) * corr_shift
  unsafe_execute!(FT⁻¹.p, f2, out)
  return out[1:lingrid.L]
end

"""
Hilbert transformation of a function defined on a linear grid
"""
@inline function hilbert(lingrid::LinearGrid, u::AbstractVector)
  @unpack padded, f1, out, FT⁻¹, FT, fft_1_x = lingrid.op

  unsafe_execute!(FT, _zeropad!(padded, u), f1)
  @. f1 *= fft_1_x
  unsafe_execute!(FT⁻¹.p, f1, out)
  return out[1:lingrid.L]
end

"""
Get index of y on the mesh (not rounded - i.e. allow for interpolated values)
"""
function invert_grid(lm::LinearGrid, y::Real)
  return 1 + (y - lm.xmin) / lm.dx
end
