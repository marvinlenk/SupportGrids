"""
    LinearGridOps(points::Vector{T}, complex_op::Bool;
                  integration_method::Symbol=:trapz,
                  δ::T=-(extrema(points)...)/(length(points)-1)) where T

Generates the necessary tools for operations on the `LinearGrid` with `points` of
discretization `δ`. For `integration_method` options, see [`integral_weights`](@ref).

# Fields
- `weights`: Integration weights.
- `ft`, `ft⁻¹`: FFT and inverse FFT plans by [`FFTW`](@ref).
- `frequencies`: Points in Fourier space.
- `padded`: Work array for padded vector / FFT input.
- `ft_work1`, `ft_work2`, `ft⁻¹_work`: Work array for (inverse) FFT output.
- `conv_shift`, `xcorr_shift`: FFT Phase correction arrays due to frequency shift.
- `fft_x⁻¹`: Fourier transform of `1/x` used for the Hilbert transform.

See also [`LinearGridOps`](@ref).
"""
struct LinearGridOps{T} <: AbstractGridOps{T}
  weights::Vector{T}
  ft::Plan
  ft⁻¹::ScaledPlan
  frequencies::Vector{Complex{T}}
  padded::Vector{<:Union{T, Complex{T}}}
  ft_work1::Vector{Complex{T}}
  ft_work2::Vector{Complex{T}}
  ft⁻¹_work::Vector{<:Union{T, Complex{T}}}
  conv_shift::Vector{Complex{T}}
  xcorr_shift::Vector{Complex{T}}
  fft_x⁻¹::Vector{Complex{T}}
  
  function LinearGridOps(points::Vector{T}, complex_op::Bool;
      integration_method::Symbol=:trapz, δ::T=-(extrema(points)...)/(length(points)-1)
      ) where T
    weights = integration_weights(points, integration_method)
    padded_len = nextpow(2, 2*length(points)-1) # double the lebgth and fill to the next power of 2
    if T <: BigFloat len_padded -= 1 end # Why?
    
    padded = zeros(complex_op ? Complex{T} : T, padded_len)
    ft⁻¹_work = similar(padded)
    ft = complex_op ? plan_fft(ft⁻¹_work) : plan_rfft(ft⁻¹_work)
    ft_work1 = zeros(Complex{T}, ft.osz)
    ft_work2 = zeros(Complex{T}, ft.osz)
    ft⁻¹ = complex_op ? plan_ifft(ft_work1) : plan_irfft(ft_work1, padded_len)
    
    frequencies = complex_op ? fftfreq(padded_len) : rfftfreq(padded_len)
    ωₛ = points[1] * 2π * frequencies / δ # Phases due to the original domain offset
    
    # NOTE: This is normalized (δ and scale)
    conv_shift = exp.(-im * ωₛ) * δ * ft⁻¹.scale
    xcorr_shift = conj.(conv_shift)
    fft_x⁻¹ = -im * sign.(frequencies) * ft⁻¹.scale
    
    new{T}(weights,ft,ft⁻¹,frequencies,padded,ft_work1,ft_work2,ft⁻¹_work,
      conv_shift,xcorr_shift,fft_x⁻¹)
  end
end

"""
    LinearGrid(xmin::Real, xmax::Real, len::Int; complex_op=false, T::Type=Float64)

Generates an equidistant grid from `xmin` to `xmax` with `len` points. If operations on
complex arrays are expected, `complex_op` should be set to `true`.

# Fields
- `xmin`, `xmax`, `len` defined above.
- `points`: Vector of grid points `pᵢ`.
- `δ`: Discretisation parameter of `points`, i.e. `pᵢ₊₁ = pᵢ + δ`.
- `op`: Tools for operations on this `LinearGrid`.

See also [`LinearGridOps`](@ref).
"""
struct LinearGrid{T} <: SupportGrid{T}
  xmin::T               # Left bound
  xmax::T               # Right bound
  len::Int              # Number of points
  points::Vector{T}     # Array of grid points
  δ::T                  # Frequency discretisation
  op::LinearGridOps{T}

  function LinearGrid(xmin::Real, xmax::Real, len::Int;
      complex_op::Bool=false, T::Type=Float64
    )
    indices = 0:(len-1)
    
    δ = (xmax - xmin) / (len - 1)
    points = δ .* collect(indices) .+ xmin
    points[end] = xmax
    
    new{T}(xmin, xmax, len, points, δ, LinearGridOps(points, complex_op; δ))
  end
  #
  LinearGrid(::Type{T}, args...; kwargs...) where T = LinearGrid(args...; T, kwargs...)
end

"""
Convolution of 2 functions defined on a linear grid
"""
function convolution(grid::LinearGrid, u::AbstractVector, v::AbstractVector)
  (; padded, ft_work1, ft_work2, ft⁻¹_work, ft, ft⁻¹, conv_shift) = grid.op
  
  unsafe_execute!(ft, _zeropad!(padded, u), ft_work1)
  unsafe_execute!(ft, _zeropad!(padded, v), ft_work2)
  @. ft_work2 *= ft_work1 * conv_shift
  unsafe_execute!(FT⁻¹.p, ft_work2, out)
  return out[1:lingrid.l]
end

"""
Cross-Correlation of 2 functions defined on a linear grid
"""
function crosscorrelation(grid::LinearGrid, u::AbstractVector, v::AbstractVector)
  (; padded, ft_work1, ft_work2, ft⁻¹_work, ft, ft⁻¹, corr_shift) = grid.op
  
  unsafe_execute!(ft, _zeropad!(padded, u), ft_work1)
  unsafe_execute!(ft, _zeropad!(padded, v), ft_work2)
  @. ft_work2 *= conj(ft_work1) * corr_shift
  unsafe_execute!(ft⁻¹.p, ft_work2, ft⁻¹_work)
  return out[1:grid.l]
end

"""
Hilbert transformation of a function defined on a linear grid
"""
@inline function hilbert(grid::LinearGrid, u::AbstractVector)
  (; padded, ft_work1, ft⁻¹_work, ft, ft⁻¹, fft_x⁻¹) = grid.op
  
  unsafe_execute!(FT, _zeropad!(padded, u), f1)
  @. f1 *= fft_1_x
  unsafe_execute!(FT⁻¹.p, f1, out)
  return out[1:grid.L]
end

"""
Get index of y on the mesh (not rounded - i.e. allow for interpolated values)
"""
function invert_grid(grid::LinearGrid, y::Real)
  return 1 + (y - grid.xmin) / grid.δ
end
