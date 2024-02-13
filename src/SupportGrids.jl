module SupportGrids

############
# Packages #
############
using LinearAlgebra
using DSP
using RecipesBase

##########
# Export #
##########
export
  AbstractMesh,
  AbstractNonlinearMesh,
  AbstractMeshOps,
  CompositeGrid,
  ExpGrid,
  ExpTanGrid,
  ExpExpGrid,
  InvLogGrid,
  InvLogTanGrid,
  LinearGrid,
  LogGrid,
  LogTanGrid,
  SqrtGrid,
  TanGrid,
  
  invert_grid,
  derivative,
  integrate,
  convolution,
  conv,
  crosscorrelation,
  xcorr,
  hilbert,
  
  print_gridparams

#############
# Abstracts #
#############
abstract type SupportGrid{T<:Any} end
abstract type AbstractNonlinearGrid{T<:Any} <: SupportGrid{T} end
abstract type AbstractGridOps{T<:Any} end

############
# Includes #
############
include("grid_shared.jl")
include("nonlinear_shared.jl")

include("composite_grid.jl")
include("exp_grid.jl")
include("exp_tan_grid.jl")
include("expexp_grid.jl")
include("invlog_grid.jl")
include("invlog_tan_grid.jl")
include("linear_grid.jl")
include("log_grid.jl")
include("log_tan_grid.jl")
include("sqrt_grid.jl")
include("tan_grid.jl")

####################
# Plotting recipes #
####################
@recipe f(::Type{T}, grid::T) where T <: SupportGrid = grid.Grid

end
