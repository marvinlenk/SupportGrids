"""
    LinearGridOps(points::Vector{T}, complex_op::Bool;
                  integration_method::Symbol=:trapz,
                  d::T=-(extrema(points)...)/(length(points)-1)) where T

Generates the necessary tools for operations on the `LinearGrid` with `points` of
discretization `d`. For `integration_method` options, see [`integral_weights`](@ref).

# Fields
- `weights`: Integration weights.
- `ft_p`, `bft_p`: FFT and backwards FFT plans by [`FFTW`](@ref).
- `frequencies`: Points in Fourier space.
- `padded`: Work array for padded vector / FFT input.
- `ft_work1`, `ft_work2`, `ft⁻¹_work`: Work array for (inverse) FFT output.
- `fft_x⁻¹`: Fourier transform of `1/x` used for the Hilbert transform.
- `ft_arr`: Array needed for the Fourier transforms.

See also [`LinearGridOps`](@ref).
"""

struct LinearGridOps{T} <: AbstractGridOps{T}
    weights::Vector{T}
    ft_p::Plan
    bft_p::Plan
    frequencies::Vector{T}
    padded::Vector{<:Union{T,Complex{T}}}
    ft_work1::Vector{Complex{T}}
    ft_work2::Vector{Complex{T}}
    bft_work::Vector{<:Union{T,Complex{T}}}
    fft_x⁻¹::Vector{Complex{T}}
    ft_arr::Vector{Int}
    data_range::UnitRange{Int}

    function LinearGridOps(points::Vector{T}, complex_op::Bool;
        integration_method::Symbol=:trapz, d::T=abs(+(extrema(points)...))
    ) where {T}

        len = length(points)

        weights = integration_weights(points, integration_method)

        # double the length and fill to the next power of 2
        padded_len = nextpow(2, 2 * len - 1)

        padded = zeros(complex_op ? Complex{T} : T, padded_len)
        bft_work = similar(padded)

        ft_p = complex_op ? plan_fft(bft_work) : plan_rfft(bft_work)

        ft_work1 = zeros(Complex{T}, ft_p.osz)
        ft_work2 = zeros(Complex{T}, ft_p.osz)

        bft_p = complex_op ? plan_bfft(ft_work1) : plan_brfft(ft_work1, padded_len)

        dk = 2π / (padded_len * d)

        ft_points = dk .* (collect(0:(padded_len-1)) .- padded_len ÷ 2)

        ft_arr = [(-1)^(i - 1) for (i, _) in enumerate(ft_work1)]

        fft_x⁻¹ = im * sign.(ft_points)

        data_range = len÷2+1:len+len÷2

        new{T}(weights, ft_p, bft_p, ft_points, padded, ft_work1, ft_work2, bft_work,
            fft_x⁻¹, ft_arr, data_range)
    end
end

"""
    LinearGrid(grid_min::Real, len::Int; complex_op=true, T::Type=Float64)

Generates a symmetric equidistant grid from `grid_min` to `-grid_min` with `len` points.
If operations on real arrays are expected, `complex_op` should be set to `false`.

# Fields
- `grid_min`, `len` defined above.
- `points`: Vector of grid points `pᵢ`.
- `d`: Discretisation parameter of `points`, i.e. `pᵢ₊₁ = pᵢ + d`.
- `op`: Tools for operations on this `LinearGrid`.

See also [`LinearGridOps`](@ref).
"""
struct LinearGrid{T} <: SupportGrid{T}
    grid_min::T            # Left bound
    len::Int              # Number of points
    points::Vector{T}     # Array of grid points
    d::T                  # Discretisation parameter
    op::LinearGridOps{T} # Tools for FFT operations on this grid

    function LinearGrid(grid_min::Real, len::Int;
        complex_op::Bool=true, T::Type=Float64
    )

        indices = 0:(len-1)

        d = 2abs(grid_min) / len

        # Symmetric grid around 0, note that the last point is not included
        points = d .* (collect(indices) .- len ÷ 2)

        new{T}(grid_min, len, points, d, LinearGridOps(points, complex_op; d))
    end
    #
    LinearGrid(::Type{T}, args...; kwargs...) where {T} = LinearGrid(args...; T, kwargs...)
end

"""
Apply FFT of grid to input
"""
function ft!(grid::LinearGrid, out::AbstractVector, u::AbstractVector)
    return fftshift(bfft(u)) .* grid.op.ft_arr[1:length(u)] * grid.d
end

ft(grid::LinearGrid, u::AbstractVector) = ft!(grid, zero(u), copy(u))

"""
Internal function that works on padded arrays
"""
function ift!(grid::LinearGrid, out::AbstractVector, u::AbstractVector)
    return fftshift(fft(u)) .* grid.op.ft_arr[1:length(u)] * grid.d / 2π
end

"""
Apply inverse FFT of grid to input
"""

ift(grid::LinearGrid, u::AbstractVector) = ift!(grid, zero(u), copy(u))

"""
Internal function that works on padded arrays
"""
function _convolution!(grid::LinearGrid, out::AbstractVector,
    u::AbstractVector, v::AbstractVector)
    (; padded, ft_work1, ft_work2, ft_p, bft_p, ft_arr) = grid.op

    assert_applicable(bft_p, ft_work2, out)
    unsafe_execute!(ft_p, _zeropad!(padded, u), ft_work1)
    unsafe_execute!(ft_p, _zeropad!(padded, v), ft_work2)
    @. ft_work2 *= ft_work1 * ft_arr

    unsafe_execute!(bft_p, ft_work2, out)

    out .*= grid.d / length(padded)

    return out
end

"""
Convolution - takes non-padded arrays as input (and output)
"""
convolution!(grid::LinearGrid, out::AbstractVector,
    u::AbstractVector, v::AbstractVector
) = out .= @views _convolution!(grid, grid.op.bft_work, u, v)[grid.op.data_range]

convolution(grid::LinearGrid, u::AbstractVector, v::AbstractVector
) = convolution!(grid, zeros(eltype(grid.op.padded), length(u)), u, v)

"""
Internal function that works on padded arrays
"""
function _crosscorrelation!(grid::LinearGrid, out::AbstractVector,
    u::AbstractVector, v::AbstractVector)
    (; padded, ft_work1, ft_work2, ft_p, bft_p, ft_arr) = grid.op

    assert_applicable(bft_p, ft_work2, out)
    unsafe_execute!(ft_p, _zeropad!(padded, u), ft_work1)
    unsafe_execute!(ft_p, _zeropad!(padded, v), ft_work2)

    @. ft_work2 *= conj(ft_work1) * ft_arr
    unsafe_execute!(bft_p, ft_work2, out)


    out .*= grid.d / length(padded)

    return out
end

"""
Cross-correlation - takes non-padded arrays as input (and output)
"""
crosscorrelation!(grid::LinearGrid, out::AbstractVector,
    u::AbstractVector, v::AbstractVector
) = out .= @views _crosscorrelation!(grid, grid.op.bft_work, u, v)[grid.op.data_range]

crosscorrelation(grid::LinearGrid, u::AbstractVector, v::AbstractVector
) = crosscorrelation!(grid, zeros(eltype(grid.op.padded), length(u)), u, v)

"""
Internal function that works on padded arrays
"""
function _hilbert!(grid::LinearGrid, out::AbstractVector, u::AbstractVector)
    (; padded, ft_work1, ft_p, bft_p, fft_x⁻¹) = grid.op

    assert_applicable(bft_p, ft_work1, out)
    unsafe_execute!(ft_p, _zeropad!(padded, u), ft_work1)
    @. ft_work1 *= fft_x⁻¹
    unsafe_execute!(bft_p, ft_work1, out)

    out .*= 1 / length(padded)

    return out
end

"""
Hilbert transform - takes non-padded arrays as input (and output)
"""
hilbert!(grid::LinearGrid, out::AbstractVector, u::AbstractVector
) = out .= @views _hilbert!(grid, grid.op.bft_work, u)[grid.op.data_range]

hilbert(grid::LinearGrid, u::AbstractVector) = hilbert!(grid, zero(u), u)

"""
Get index of y on the mesh (not rounded - i.e. allow for interpolated values)
"""
function invert_grid(grid::LinearGrid, y::Real)
    return 1 + (y - grid.xmin) / grid.d
end
