module SupportGrids

############
# Packages #
############
using RecipesBase: @recipe
using UnPack: @unpack
import Base.*, Base.+, Base.-

using DSP: _zeropad!
using DSP.FFTW: unsafe_execute!, fftfreq, rfftfreq
using DSP.FFTW: plan_fft, plan_rfft, plan_ifft, plan_irfft
using DSP.FFTW.AbstractFFTs: Plan, ScaledPlan

using LinearAlgebra: ⋅
import LinearAlgebra.dot

##########
# Export #
##########
export
  AbstractMesh,
  AbstractNonlinearMesh,
  AbstractMeshOps,
  # CompositeGrid,
  # ExpGrid,
  # ExpTanGrid,
  # ExpExpGrid,
  # InvLogGrid,
  # InvLogTanGrid,
  LinearGrid,
  # LogGrid,
  # LogTanGrid,
  # SqrtGrid,
  # TanGrid,
  
  derivative,
  # derivative!,
  integrate,
  integrate!,
  convolution,
  # convolution!,
  crosscorrelation,
  # crosscorrelation!,
  hilbert,
  # hilbert!,
  
  invert_grid,
  print_gridparams

#############
# Abstracts #
#############
abstract type SupportGrid{T<:Any} end
abstract type AbstractNonlinearGrid{T<:Any} <: SupportGrid{T} end
abstract type AbstractGridOps{T<:Any} end

###############################
# Includes , docs and aliases #
###############################
@doc "Alias: `∫ = integrate`" integrate
@doc "Alias: `∫! = integrate!`" integrate!
@doc "Alias: `conv = convolution`" convolution
@doc "Alias: `conv = convolution!`" convolution!
@doc "Alias: `xcorr = crosscorrelation`" crosscorrelation

include("grid_shared.jl")
# include("nonlinear_shared.jl")

# include("grids/composite_grid.jl")
# include("grids/exp_grid.jl")
# include("grids/exp_tan_grid.jl")
# include("grids/expexp_grid.jl")
# include("grids/invlog_grid.jl")
# include("grids/invlog_tan_grid.jl")
include("grids/linear_grid.jl")
# include("grids/log_grid.jl")
# include("grids/log_tan_grid.jl")
# include("grids/sqrt_grid.jl")
# include("grids/tan_grid.jl")

const ∫ = integrate
const ∫! = integrate!
export ∫, ∫!

const conv = convolution
# const conv! = convolution!
export conv#, conv!

const xcorr = crosscorrelation
# const xcorr! = crosscorrelation!
export xcorr#, xcorr!


####################
# Plotting recipes #
####################
@recipe f(::Type{T}, grid::T) where T <: SupportGrid = grid.Grid

end
