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
- `ft_shift`, `ft⁻¹_shift`: FFT Phase and scale correction due to grid offset.
- `conv_shift`: Same as above with added scaling for `conv`.
- `fft_x⁻¹`: Fourier transform of `1/x` used for the Hilbert transform.

See also [`LinearGridOps`](@ref).
"""
struct LinearGridOps{T} <: AbstractGridOps{T}
  weights::Vector{T}
  ft::Plan
  ft⁻¹::ScaledPlan
  frequencies::Vector{T}
  padded::Vector{<:Union{T, Complex{T}}}
  ft_work1::Vector{Complex{T}}
  ft_work2::Vector{Complex{T}}
  ft⁻¹_work::Vector{<:Union{T, Complex{T}}}
  ft_shift::Vector{Complex{T}}
  ft⁻¹_shift::Vector{Complex{T}}
  conv_shift::Vector{Complex{T}}
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
    
    # Includes scale of unitary FT
    ft_shift = exp.(-im * ωₛ) * δ
    ft⁻¹_shift = exp.(im * ωₛ) * ft⁻¹.scale / δ
    conv_shift = ft_shift * ft⁻¹.scale
    fft_x⁻¹ = -im * sign.(frequencies) * ft⁻¹.scale
    
    new{T}(weights,ft,ft⁻¹,frequencies,padded,ft_work1,ft_work2,ft⁻¹_work,
      ft_shift,ft⁻¹_shift,conv_shift,fft_x⁻¹)
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
Apply FFT of grid to input
"""
function fft!(grid::LinearGrid, out::AbstractVector, u::AbstractVector)
  (; padded, ft, ft⁻¹, ft_shift) = grid.op
  
  assert_applicable(ft, padded, out)
  unsafe_execute!(ft, _zeropad!(padded, u), out)
  @. out *= ft_shift
  
  return out
end

fft(grid::LinearGrid, u::AbstractVector) = fft!(grid, zero(grid.op.ft_work1), u)

"""
Internal function that works on padded arrays
"""
function _ifft!(grid::LinearGrid, out::AbstractVector, u::AbstractVector)
  (; ft_work1, ft⁻¹, ft⁻¹_shift) = grid.op
  
  assert_applicable(ft⁻¹.p, ft_work1, out)
  _zeropad!(ft_work1, u; fillzero=true)
  @. ft_work1 *= ft⁻¹_shift
  unsafe_execute!(ft⁻¹.p, ft_work1, out)
  
  return out
end

"""
Apply inverse FFT of grid to input
"""
ifft!(grid::LinearGrid, out::AbstractVector, u::AbstractVector
  ) = out .= @views _ifft!(grid, grid.op.ft⁻¹_work, u)[1:grid.len]

ifft(grid::LinearGrid, u::AbstractVector
  ) = ifft!(grid, zeros(eltype(grid.op.ft⁻¹_work), grid.len), u)

"""
Internal function that works on padded arrays
"""
function _convolution!(grid::LinearGrid, out::AbstractVector,
    u::AbstractVector,v::AbstractVector)
  (; padded, ft_work1, ft_work2, ft, ft⁻¹, conv_shift) = grid.op
  
  assert_applicable(ft⁻¹.p, ft_work2, out)
  unsafe_execute!(ft, _zeropad!(padded, u), ft_work1)
  unsafe_execute!(ft, _zeropad!(padded, v), ft_work2)
  @. ft_work2 *= ft_work1 * conv_shift
  unsafe_execute!(ft⁻¹.p, ft_work2, out)
  
  return out
end

"""
Convolution - takes non-padded arrays as input (and output)
"""
convolution!(grid::LinearGrid, out::AbstractVector,
  u::AbstractVector,v::AbstractVector
  ) = out .= @views _convolution!(grid, grid.op.ft⁻¹_work, u, v)[1:grid.len]

convolution(grid::LinearGrid, u::AbstractVector, v::AbstractVector
  ) = convolution!(grid, zero(u), u, v)

"""
Internal function that works on padded arrays
"""
function _crosscorrelation!(grid::LinearGrid, out::AbstractVector,
    u::AbstractVector, v::AbstractVector)
  (; padded, ft_work1, ft_work2, ft, ft⁻¹, corr_shift) = grid.op
  
  assert_applicable(ft⁻¹.p, ft_work2, out)
  unsafe_execute!(ft, _zeropad!(padded, u), ft_work1)
  unsafe_execute!(ft, _zeropad!(padded, v), ft_work2)
  @. ft_work2 *= conj(ft_work1 * corr_shift)
  unsafe_execute!(ft⁻¹.p, ft_work2, out)
  
  return out
end

"""
Cross-correlation - takes non-padded arrays as input (and output)
"""
crosscorrelation!(grid::LinearGrid, out::AbstractVector,
  u::AbstractVector, v::AbstractVector
  ) = out .= @views _crosscorrelation!(grid, grid.op.ft⁻¹_work, u, v)[1:grid.len]

crosscorrelation(grid::LinearGrid, u::AbstractVector, v::AbstractVector
  ) = crosscorrelation!(grid, zero(u), u, v)

"""
Internal function that works on padded arrays
"""
function _hilbert!(grid::LinearGrid, out::AbstractVector, u::AbstractVector)
  (; padded, ft_work1, ft, ft⁻¹, fft_x⁻¹) = grid.op
  
  assert_applicable(ft⁻¹.p, ft_work1, out)
  unsafe_execute!(ft, _zeropad!(padded, u), ft_work1)
  @. ft_work1 *= fft_x⁻¹
  unsafe_execute!(ft⁻¹.p, ft_work1, out)
  
  return out
end

"""
Hilbert transform - takes non-padded arrays as input (and output)
"""
hilbert!(grid::LinearGrid, out::AbstractVector, u::AbstractVector
  ) = out .= @views _hilbert!(grid, grid.op.ft⁻¹_work, u)[1:grid.len]

hilbert(grid::LinearGrid, u::AbstractVector) = hilbert!(grid, zero(u), u)

"""
Get index of y on the mesh (not rounded - i.e. allow for interpolated values)
"""
function invert_grid(grid::LinearGrid, y::Real)
  return 1 + (y - grid.xmin) / grid.δ
end
