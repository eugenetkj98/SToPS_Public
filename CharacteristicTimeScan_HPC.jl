include("ChaoticSystems.jl")
include("HomologyAnalysis.jl")
include("GrahamScan.jl")

"""
Latest Update Date: 29/11/2022
Author: Eugene Tan, UWA Complex Systems Group

This script contains functions to run and calculate the SToPS significance score from an input experimental time series
Several hyperparameters are also given to be tweaked.

SToPS() is the function to run the analyses.
"""

using FileIO
using JLD2
using DataFrames
using CSV
using Clustering
using DynamicalSystems
using MultivariateStats
using ProgressBars

# Code to calculate the persistence cutoff value required to remove topological noise
function cutoff_calc(S, n; mode = "mean")
    if mode == "mean"
        ϵ = mean(sqrt.(sum((S[2:n:end,:]-S[1:n:end-1,:]).^2, dims = 2)))
    else
        ϵ = quantile(sqrt.(sum((S[2:n:end,:]-S[1:n:end-1,:]).^2, dims = 2))[:], 0.9)
    end
    return ϵ
end

# Calculate persistence entropy from a given diagram
function persistence_entropy(PD)

    lifetimes = zeros(length(PD))
    for i in 1:length(PD)
        lifetimes[i] = PD[i].death-PD[i].birth
    end
    p = lifetimes./sum(lifetimes)

    return -sum(p.*log2.(p))
end

# Calculates the circularity/roundness [0,1] of the homology generator based on PCA variances projected onto the 2D flattest plane
function circularity(hole_points)
    PCA_model = fit(PCA, hole_points'; maxoutdim=2, pratio = 1)
    λ_1, λ_2 = principalvars(PCA_model)
    α = λ_2/λ_1
    return α
end

# Given a sequence of clockwise ordered 2d points (nx2), calculate the polygon area using the determinants method
function polygon_area(ordered_points)
    determinants = zeros(size(ordered_points)[1])

    for i in 1:length(determinants)
        if i == length(determinants)
            determinants[i] = det(ordered_points[[end,1],:]')
        else
            determinants[i] = det(ordered_points[i:i+1,:]')
        end
    end

    A_homology = sum(determinants)/2
end

""" given the following:
(1) Sequence of points ordered clockwise forming the polygon of the homology generator
(2) Point cloud from which homology was calculated
Calculates the Efficiency [0,1] (ratio to which the homology hole size covers the point cloud)
hole_points and points (points cloud) are NxD matrices
"""

function efficiency(hole_points, points)
    # Project homology generator and point cloud into 2d plane that best preserves hole
    M = fit(PCA, hole_points'; maxoutdim=2, pratio = 1)
    P = M.proj # Projection Matrix
    μ = M.mean # Mean values
    y_homology = transpose(P' *(Matrix(hole_points').-μ)) # Homology polygon in transformed space
    y_pc = transpose(P' *(Matrix(points').-μ)) # Point cloud in transformed space


    # Calculate area of homology generator polygon
    ordered_points_homology = y_homology[end:-1:1,:]
    A_homology = polygon_area(ordered_points_homology)

    # Calculate area of 2D convex polygon using Graham scan
    convex_hull_boundary = graham(y_pc)["Boundary"]
    A_pc = polygon_area(convex_hull_boundary)

    # Final value for efficiency
    β = abs.(A_homology/A_pc)

    return β
end

function calc_score_helper_function(n, p, ϵ, idx, S)
    # Extract Starting Position
    points = deepcopy(@view S[idx:n:idx+p,:])
    pc = pointify(points')

    # Calculate Homology
    # PD = ripserer(pc, dim_max = 1, reps = true, cutoff = ϵ)[2]
    PD = ripserer(pc, dim_max = 1, reps = true)[2]

    # Temporary output storage
    life_temp_output = 0
    α_temp_output = 0
    β_temp_output = 0

    # Check to make sure 1D Homology Exists, otherwise skip
    if length(PD) != 0
        # Find the most persistent feature
        most_persistent_co = PD[end]

        # Extract persistence
        life_temp_output = most_persistent_co.death-most_persistent_co.birth

        # Calculate corresponding persistent homology
        # p_ents[j] = persistence_entropy(PD)

        # Extract most polygon points persistent homology generator
        reconstructed_at_birth = reconstruct_cycle(PD.filtration, most_persistent_co)
        vertex = vertices.(reconstructed_at_birth)

        hole_index = zeros(Int,length(reconstructed_at_birth))
        for k in 1:length(hole_index)
            # Need to deal with ordering of points in representative cycles
            if k!= length(hole_index)
                if vertex[k][1] in vertex[k+1]
                    hole_index[k] = deepcopy(vertex[k][1])
                else
                    hole_index[k] = deepcopy(vertex[k][2])
                end
            else
                if vertex[k][1] in vertex[1]
                    hole_index[k] = deepcopy(vertex[k][1])
                else
                    hole_index[k] = deepcopy(vertex[k][2])
                end
            end
        end
        hole_points = deepcopy(@view points[hole_index,:])

        # Calculate Circularity and Eccentricity

        # if α*β > score_threshold
        α_temp_output = circularity(hole_points)
        β_temp_output = efficiency(hole_points, points)
        # end
    end
    return life_temp_output, α_temp_output, β_temp_output
end

"""
Calculate SToPS Characteristic times from an input univariate time series x.
Also calculates some non-uniform delay lags from automated embedding methods PECUZAL, MDOP.
Outputs results into a jld2 file with name "filename".
"""
function SToPS(x_input; filename = "output.jld2")
    # %% Hyperparameters
    τ_lags = 1:1:150 # Range of lag values to test
    window_scales = 4:4 # Size of window to select strand (set at 4τ)
    SAMPLE_SIZE = 500 # Number of strands to sample
    MAX_STRAND_LENGTH = 250 # Approximate maximum strand length

    # %% Extract and pre-process data
    x = deepcopy(x_input)
    
    # %% Normalise data to unit interval
    x = (x .-minimum(x))./(maximum(x)-minimum(x))

    # %% Add noise and scaling to fix precision issues
    x = 5*x .+ 0.015*rand(Float64, size(x))

    pers = zeros(length(τ_lags), SAMPLE_SIZE, length(window_scales))
    α_val = zeros(length(τ_lags), SAMPLE_SIZE, length(window_scales))
    β_val = zeros(length(τ_lags), SAMPLE_SIZE, length(window_scales))

    progress_counter = zeros(length(τ_lags))

    for i in 1:length(τ_lags)

        τ = deepcopy(τ_lags[i]) # Chosen lag
        w = floor(Int,window_scales[end]*τ) # Maximum period being tested
        S = custom_embed(x, τ, 2)


        idxs = initseeds(:kmpp,S[1:size(S)[1]-w,:]', SAMPLE_SIZE)
        lifes = zeros(length(idxs), length(window_scales))
        # p_ents = zeros(length(idxs))
        α_vals = zeros(length(idxs), length(window_scales))
        β_vals = zeros(length(idxs), length(window_scales))


        for j in 1:length(idxs)
            for l in 1:length(window_scales)
                # Construct Point Cloud
                p = floor(Int,window_scales[l]*τ) # Current period being tested

                if p > MAX_STRAND_LENGTH
                    n = round(Int, p/MAX_STRAND_LENGTH)
                else
                    n = 1
                end

                # Calculate persistence cutoff for homology
                ϵ = cutoff_calc(S, n, mode = "mean")

                # Construct temporary holder variables in memory
                S_temp = deepcopy(S)

                idx = deepcopy(idxs[j])

                # Calculate component scores and assign to temporary storage
                life, α, β = calc_score_helper_function(n, p, ϵ, idx, S_temp)

                lifes[j,l] = life
                α_vals[j,l] = α
                β_vals[j,l] = β
            end
        end


        pers[i,:,:] .= deepcopy(lifes)
        α_val[i,:,:] .= deepcopy(α_vals)
        β_val[i,:,:] .= deepcopy(β_vals)

        progress_counter[i] = 1
        println("Progress: $(sum(progress_counter))/$(length(τ_lags))")
        flush(stdout)
    end

    # %% Calculate Mutual Information
    println("Calculating Mutual Information")
    flush(stdout)
    MI_values = selfmutualinfo(x,τ_lags)

    # %% Try other non-uniform embedding methods
    println("Calculating Theiler window...")
    flush(stdout)
    theiler = estimate_delay(x, "mi_min") # estimate a Theiler window

    println("Calculating PECUZAL embedding...")
    flush(stdout)
    # Pecuzal embedding
    τ_pecuzal =  pecuzal_embedding(x, τs = τ_lags, w = theiler, econ = true, max_cycles = 5)[2]

    println("Calculating MDOP embedding...")
    flush(stdout)

    # Maximising Derivatives on Projection
    τ_mdop =  mdop_embedding(x, τs = τ_lags, w = theiler, max_num_of_cycles = 5)[2]

    # %% Save data
    data = Dict("timeseries" => x, "dt" => dt, "lags" => τ_lags, "strand_scale" => window_scales,
                "persistence" => pers, "circularity" => α_val, "efficiency" => β_val,
                "pecuzal_embed" => τ_pecuzal, "mdop_embed" => τ_mdop, "MI" => MI_values)

    # %%
    println("Saving Data...")
    flush(stdout)

    save(filename, data)

    println("Saved.")
    flush(stdout)
end