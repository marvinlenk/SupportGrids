"""
Everything
"""
function expexpgrid(ω::Real, dout::Real, L::Int; sclose::Real=1e-9)
#   println(ω, " ", dout, " ", L)
  b = log(sclose)
  if abs(ω) < sclose*10
    L2 = Int(L/2)
    a = (log(dout) - b) / (L2 - 1)
    lingrid = UnitRange(0, L2 - 1)
    posgrid = expgrid.(lingrid, a, b)
    posgrid[end] = dout 
    return [-reverse(posgrid); posgrid]
  else
    γ = (log(abs(ω)/2) - b)/(log(dout) - b)
    # Calculate points on inner and outer grids
    Lin = round(Int, (1 + γ*(L-2))/(1 + γ))
    Lin += Lin % 2 == 1 ? 1 : 0
    Lout = L - Lin
    Lin2 = Int(Lin/2)
    Lout2 = Int(Lout/2)
    # calculate inner and outer scaling parameters
    ain = (log(abs(ω)/2) - b) * 2/(Lin-1)
    aout = (log(dout) - b) / (Lout2-1)
    posgrid_in = broadcast(x->expgrid(x, ain, b), 0:(Lin2-1))
    posgrid_out = broadcast(x->expgrid(x, aout, b), 0:(Lout2-1))
    mesh = [-reverse(posgrid_out); posgrid_in ; abs(ω) .- reverse(posgrid_in); abs(ω) .+ posgrid_out]
    mesh[1] = -dout
    mesh[end] = abs(ω) + dout
    return ω < 0 ? -reverse(mesh) : mesh
  end
end

"""
  widths are HWHM (tan meshes are not optimal but will have to make do)
  Tan meshes should be six times the width of the lorentzians (width parameter)
  
  expL_minfrac minimum fraction of points for the exponential functions
"""
function helpermesh!(mesh::AbstractVector, ω::Real, dout::Real;
    peaks=[0], widths=10dout * one.(peaks), sclose=1e-9, expL_minfrac=1/2)
  #Make sure peaks are sorted
  if !issorted(peaks)
    perm = sortperm(peaks)
    peaks, widths = peaks[perm], widths[perm]
  end
  
  L = length(mesh)
  xmin_h = minimum([0,ω])
  xmax_h = maximum([0,ω])
  xmin = xmin_h - dout
  xmax = xmax_h + dout
  #First check if peaks lie between the two Exp grid centers
  peaks_in_range = xmin_h .< peaks .< xmax_h
  #Then check for peaks that are not too close to the centers (i.e. need more resolution)
  peaks_resolved = abs.(peaks .- xmin_h) .> widths/10
  peaks_resolved .*= abs.(peaks .- xmax_h) .> widths/10
  #Those are the peaks that then will get additional mesh points
  peaks_considered = peaks_resolved .* peaks_in_range
  
  if abs(ω) < sclose*10 || abs(ω) < dout || all(peaks_considered .== false)
    #If width is very small or no peaks in range, resort to expexpgrid
    mesh .= expexpgrid(ω, dout, L; sclose)
  else
    #Otherwise, try to fit in all peaks that are in range
    #Note that in this section we always have at least one peak
    if count(peaks_considered) == 1
      peaks_eff = peaks[peaks_considered] #only peaks that are relevant
      widths_eff = widths[peaks_considered]
    else
      #For multiple peaks we need to fill gaps
      peaks_eff = zeros(Float64,2 * count(peaks_considered) - 1)
      widths_eff = zeros(Float64,2 * count(peaks_considered) - 1)
      peaks_eff[1:2:end] .= peaks[peaks_considered] #only peaks that are relevant
      widths_eff[1:2:end] .= widths[peaks_considered]
      # determine filling by weighted average (weights are interchanged to get repulsion)
      wr,wl = widths_eff[3:2:end], widths_eff[1:2:end-2]
      pr,pl = peaks_eff[3:2:end], peaks_eff[1:2:end-2]
      @. peaks_eff[2:2:end] = (wl * pr + wr * pl) / (wl + wr)
      @. widths_eff[2:2:end] = ((pr - wr) - (pl + wl)) / 2
    end
    numpeaks_eff = length(peaks_eff)
    
    #Define crossovers. First is minimum of ω and 0, last is maximum of those
    crossover = zeros(Float64, numpeaks_eff+3)
    crossover[1] = minimum([ω, 0])
    crossover[end] = maximum([ω, 0])
    
    #Iterate over possible crossovers
    for i in 2:length(crossover)-1
      #First and last crossover need special treatment
      #due to adjacent ExpMeshes
      if i == 2
        #ExpMesh to the left, first peak to the right
        leftpeak = crossover[1]
        leftwidth = dout
        rightpeak = peaks_eff[1]
        rightwidth = widths_eff[1]
      elseif i == length(crossover)-1
        #Last peak to the left, ExpMesh to the right
        leftpeak = peaks_eff[end]
        leftwidth = widths_eff[end]
        rightpeak = crossover[end]
        rightwidth = dout
      else
        #Peaks to the left and the right
        leftpeak = peaks_eff[i-2]
        leftwidth = widths_eff[i-2]
        rightpeak = peaks_eff[i-1]
        rightwidth = widths_eff[i-1]
      end
      #The crossover point needs to be between the left and right peak.
      #The tanh determines the crossover position depending on the relative
      #widths of the peaks
      crossover[i] = tanhmediate(leftpeak,rightpeak,(rightwidth - leftwidth)*10)
      #Make sure the empty spaces are not too large
      if i == 2 && rightpeak - crossover[i] > 1e2 * rightwidth
        crossover[i] = rightpeak - 1e2 * rightwidth
      elseif i == length(crossover)-1 && crossover[i] - leftpeak > 1e2 * leftwidth
        crossover[i] = leftpeak + 1e2 * leftwidth
      end
    end
    
    #Meshtypes are Exp for outer ones, Tan for inner ones
    meshtypes_h = [:Tan for i in 1:numpeaks_eff]
    meshtypes = [:Exp, :Exp, meshtypes_h..., :Exp, :Exp]
    
    Lfracs_h = [1 for i in 1:numpeaks_eff]
    M = 3
    Lfracs_h[1:2:end] .= M #actual peaks get more points
    #Weight N for Exp is determined by checking the sum of [4N*M, M*(numpeaks_eff-1), (numpeaks_eff-1)/2]
    #where M is a factor that suppresses the filling meshes length by 1/M.
    #N needs to be large enough to give at least expL_minfrac. This is achieved by calculating the fraction
    #4N*M / (4N*M + M*(numpeaks_eff-1) + (numpeaks_eff-1)/2) and have it at least expL_minfrac.
    N_eff = round(Int, (numpeaks_eff-1)*(M+1)/(4M * (1/expL_minfrac - 1)), RoundUp)
    LfracExp = M * maximum([N_eff, 1])
    Lfracs = [LfracExp, LfracExp, Lfracs_h..., LfracExp, LfracExp]
    
    # generate Larr and make sure it reproduces the correct L
    Larr = round.(Int, Lfracs .* (L / sum(Lfracs)))
    Ldiff = L - sum(Larr)
    # the difference between L and sum(Larr) is distributed equaly to the end of Larr
    Larr[end-abs(Ldiff)+1:end] .+= sign(Ldiff)
    Larr_cumsum = [0; cumsum(Larr)]
    
    x_h = [xmin; crossover...; xmax]
    
    for (i,el) in enumerate(meshtypes)
      @views meshv = mesh[Larr_cumsum[i]+1:Larr_cumsum[i+1]]
      # all meshes except the first (the last) need an additional point to be left out for no duplicates
      additional = i in [1, length(meshtypes)] ? 0 : 1
      L_i = Larr[i] + additional
      xmin_i, xmax_i = x_h[i:i+1]
      lingrid = UnitRange(0, Larr[i] - 1)
      pref = 1
      
      if i in [1, length(meshtypes)-1]
        # first and second to last need reversed mesh and mult with -1
        lingrid = reverse(lingrid)
        pref = -1
      end
      
      if el === :Exp
        if i in [1,2]
          x0=crossover[1]
        else
          x0=crossover[end]
        end
        σ = xmax_i - xmin_i
        a, b = ExpMesh_params(σ, sclose, L_i)
        @. meshv = pref * expgrid(lingrid .+ additional, a, b) + x0
      else
        xdens=peaks_eff[i-2]
        s=widths_eff[i-2]
        a, b = TanMesh_params(xmin_i, xmax_i, L_i; xdens, s)
        @. meshv = tangrid(lingrid .+ additional, a, b, s) + xdens
      end
    end
    
    mesh[1] = xmin
    mesh[end] = xmax
  end
  
  # backup if 
  if !all(isfinite.(mesh))
    mesh .= expexpgrid(ω, dout, L; sclose)
  end
  
  return mesh
end

helpermesh(ω::Real, dout::Real, L::Int; kwargs...) = helpermesh!(zeros(Float64,L), ω, dout; kwargs...)
