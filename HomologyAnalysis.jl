using Eirene
using ProgressBars
using Ripserer
using StaticArrays
using StatsBase
using NearestNeighbors
using Clustering
using ProgressBars

"""
This code contains helper functions, used in conjunction with Ripserer,
to manipulate outputs from persistent homology analysis.
"""

# %%
# Sample Point Cloud from trajectory
function sample_PC(S; SAMPLE_SIZE = 800)
    idx = sample(1:size(S)[1],SAMPLE_SIZE, replace = false)
    pc = Array(transpose(S[idx,:]))
    return pc
end

function pointify(mat)
    points = []
    dims = size(mat)[1]
    for vec in 1:size(mat)[2]
        v = SVector{dims, Float64}(mat[:,vec])
        push!(points, v)
    end
    return points
end

function sigmoid(x; k = 10, μ = 0.5)
    return 1/(1+exp(-k*(x-μ)))
end

function truncated_sigmoid(x; k = 10, μ = 0.5, cutoff = 0.01)
    # Remove points where lifetime is considered "NOISE"
    output = 0
    if x < cutoff
        output = 0
    else
        output = 1/(1+exp(-k*(x-μ)))
    end
    return output
end

# Calculate the components required for persistent images
function analyse_homology(pc; dim = 1)
    B = barcode(eirene(pc, model = "pc", maxdim = dim), dim = dim)
    T_B  = hcat(B[:,1],B[:,2]-B[:,1])
    return T_B
end

function ripserer_homology(pc; dim = 1, mode = "pc")
    if mode == "pc"
        points = pointify(pc)
        PD = ripserer(points, dim_max = dim, verbose = false)
    else
        kNN_A = kNN_array(pc)
        PD = ripserer(kNN_A, dim_max = dim, verbose = false)
    end
    T_B_dims = []
    for dim in 1:length(PD)
        T_B  = hcat(birth.(PD[dim]),death.(PD[dim])-birth.(PD[dim]))
        T_B_dims = push!(T_B_dims, T_B)
    end
    return T_B_dims
end

# Remove any inf values during Alpha filtration
function remove_infs(birth_life_array)
        A = deepcopy(birth_life_array)
        #First occurence of Inf value
        idx = findfirst(isinf,A[:,2])
        if !isnothing(idx)
                A = A[1:(idx-1),:]
        end
        return A
end

function findScales(T_B_collection)
    collection_max_birth = zeros(length(T_B_collection))
    collection_max_death = zeros(length(T_B_collection))
    collection_max_life = zeros(length(T_B_collection))
    for i in 1:length(T_B_collection)
        samples = T_B_collection[i]
        sample_max_birth = zeros(length(samples))
        sample_max_death = zeros(length(samples))
        sample_max_life = zeros(length(samples))
        for j in 1:length(samples)
            if isempty(samples[j])
                continue
            else
                sample_max_birth[j] = maximum(samples[j][:,1])
                sample_max_death[j] = maximum(samples[j][:,2].+samples[j][:,1])
                sample_max_life[j] = maximum(samples[j][:,2])
            end
        end
        collection_max_birth[i] = maximum(sample_max_birth)
        collection_max_death[i] = maximum(sample_max_death)
        collection_max_life[i] = maximum(sample_max_life)
    end
    max_birth = maximum(collection_max_birth)
    max_death = maximum(collection_max_death)
    max_life = maximum(collection_max_life)
    return max_birth, max_death,max_life
end

function findWeights(T_B_collection, max_life)
    w_collection = []
    for i in 1:length(T_B_collection)
        w_samples = []
        for j in 1:length(T_B_collection[i])
            # w = sigmoid.(T_B_collection[i][j][:,2]./max_life)
            w = sigmoid.(T_B_collection[i][j][:,2]./max_life, k = 10, μ = 0.25).^2
            w_samples = push!(w_samples, deepcopy(w))
        end
        w_collection = push!(w_collection, deepcopy(w_samples))
    end
    return w_collection
end

function scaleHomology(T_B_collection, max_birth, max_life)
    T_B_scaled = Array{Any}(undef, length(T_B_collection))
    for i in 1:length(T_B_collection)
        scaled_sample = Array{Any}(undef, length(T_B_collection[i]))
        for j in 1:length(T_B_collection[i])
            T_B = deepcopy(T_B_collection[i][j])
            T_B[:,1] = T_B[:,1]./max_birth
            T_B[:,2] = T_B[:,2]./max_life
            scaled_sample[j] = deepcopy(T_B)
        end
        T_B_scaled[i] = deepcopy(scaled_sample)
    end
    return T_B_scaled
end

function raw_PI_images(T_B_collection, w_collection, max_birth, max_death; resolution = 250, σ = 0.05)
    println("Calculating Images...")
    τ_images = zeros(length(T_B_collection), resolution+1, resolution+1)
    Threads.@threads for i in ProgressBar(1:length(T_B_collection), leave = false)
        images = zeros(length(T_B_collection[i]), resolution+1, resolution+1)

        for j in 1:length(T_B_collection[i])
            img = PI(T_B_collection[i][j],w_collection[i][j]
                        ,X_range = (0,max_birth), Y_range = (0,max_death), resolution=resolution, σ = σ)
            images[j,:,:] = deepcopy(@views(img))
        end
        τ_images[i,:,:] = mean(images, dims = 1)[1,:,:]
    end
    return τ_images
end

function normalised_PI_images(T_B_collection, w_collection; resolution = 250, σ = 0.05)
    println("Calculating Images...")
    τ_images = zeros(length(T_B_collection), resolution+1, resolution+1)
    for i in ProgressBar(1:length(T_B_collection), leave = false)
        images = zeros(length(T_B_collection[i]), resolution+1, resolution+1)

        for j in 1:length(T_B_collection[i])
            img = PI(T_B_collection[i][j],w_collection[i][j]
                        ,X_range = (0,1), Y_range = (0,1), resolution=resolution, σ = σ)
            images[j,:,:] = deepcopy(@views(img))
        end
        τ_images[i,:,:] = mean(images, dims = 1)[1,:,:]
    end
    return τ_images
end

#Calculate Gaussian snapshots of each point on persistence diagram
function gaussian2D(X_vals, Y_vals, μ; σ=0.05)
    img = zeros(length(X_vals),length(Y_vals))

    for i in 1:length(X_vals)
        for j in 1:length(Y_vals)
            x,y = X_vals[i], Y_vals[j]
            img[i,j] = (1/(2*π*σ^2))*exp(-((x-μ[1])^2+(y-μ[2])^2)/(2*σ^2))
        end
    end
    return img
end

function gaussianEval(position, μ; σ=0.05)
    x,y = position
    output = (1/(2*π*σ^2))*exp(-((x-μ[1])^2+(y-μ[2])^2)/(2*σ^2))
    return output
end

function eval_PI_val(position, samples_T_B, weights)
    output = 0
    for i in 1:length(samples_T_B)
        for j in 1:size(samples_T_B[i])[1]
            output += weights[i][j]*gaussianEval(position, samples_T_B[i][j,:])
        end
    end
    output = output/length(samples_T_B)
    return output
end


# Calculate persistent image
function PI(T_B,w;X_range = (0,10), Y_range = (0,2.5), resolution=100, σ = 0.2)
    X_min,X_max = X_range
    Y_min,Y_max = Y_range
    X_vals = range(X_min, X_max, length = resolution+1)
    Y_vals = range(Y_min, Y_max, length = resolution+1)

    #Store image slices in array
    img_slice = zeros(length(w), length(X_vals), length(Y_vals))

    Threads.@threads for i in 1:length(w)
        μ = T_B[i,:]
        if w[i] != 0
			out = transpose(gaussian2D(X_vals, Y_vals, μ, σ=σ))
			img_slice[i,:,:] = w[i].*out
		end
    end
    return sum(img_slice, dims = 1)[1,:,:]
end

function H0_histograms(T_B_collection_0)
    bins = 0:0.01:1
    histograms = []
    max_life = []
    for i in 1:length(T_B_collection_0)
        deaths = T_B_collection_0[i]
        max_life = push!(max_life, maximum(deaths[deaths.!=Inf]))
    end
    max_life = maximum(max_life)


    for i in 1:length(T_B_collection_0)
        h = normalize(fit(Histogram, T_B_collection_0[i]./max_life, bins), mode = :pdf)
        histograms = push!(histograms, deepcopy(h))
    end
    return histograms
end



function H1_histograms(T_B_collection_1)
    bins = 0:0.01:1
    histograms = []

    max_birth, max_life = findScales(T_B_collection_1)
    w_collection = findWeights(T_B_collection_1, max_life)
    scaled_TB = scaleHomology(T_B_collection_1, max_birth, max_life)

    for i in 1:length(scaled_TB)
        lifetimes = zeros((0,2))
        for j in 1:length(scaled_TB[i])
            lifetimes = vcat(lifetimes,scaled_TB[i][j])
        end
        lifetimes = lifetimes[:,2]
        h = normalize(fit(Histogram, lifetimes, bins), mode = :pdf)
        histograms = push!(histograms, deepcopy(h))
    end

    return histograms
end

#Convert vector to unit vector
function unitvec(vec)
    mag = sqrt(sum(vec.^2))
    if mag != 0
        output = vec./mag
    else
        output = min(mag,1)*vec
    end
    return output
end


# %% PSO Feature tracking
# samples_TB is a collection of points clouds at a given time τ
# weights is a collection of weights for each point cloud at a given time τ

function PSO(samples_TB, weights; n_particles = 80, iterations = 120, dt = 0.02, r_neighbour = 0.07)

    # Influence Coefficients
    M = 0.65 # Momentum coefficient
    ϕ_c = 1 # Cognitive coefficient
    ϕ_s = 0.1 # Social coefficient
    ϕ_n = 0.5 # Neighbour coefficient
    ϕ_r = 1 # Random perturbation coefficient

    n_neighbours = 10

    # Initialise particles
    x = zeros(iterations+1,n_particles, 2) #positions
    u = zeros(size(x)) #velocities
    x[1,:,:] .= rand(n_particles,2)
    u[1,:,:] .= 2 .*(rand(n_particles,2).-0.5)

    # Calculate best positions
    indiv_best = deepcopy(x[1,:,:])
    indiv_value = zeros(n_particles)
    for i in 1:n_particles
        indiv_value[i] = eval_PI_val(indiv_best[i,:], samples_TB, weights)
    end
    global_best = x[1,1,:]
    global_value = eval_PI_val(global_best, samples_TB, weights)
    for i in 1:n_particles
        value = eval_PI_val(x[1,i,:], samples_TB, weights)
        if value > global_value
            global_best = deepcopy(x[1,i,:])
            global_value = deepcopy(value)
        end
    end

    for t in 1:iterations
        r_c = ones(n_particles)#rand(n_particles) # Random cognitive coefficient
        r_s = ones(n_particles)#rand(n_particles) # Random social coefficient
        kdtree = KDTree(transpose(x[t,:,:]))
        # idxs, dists = knn(kdtree, transpose(x[t,:,:]), n_neighbours, true)
        idxs = inrange(kdtree, transpose(x[t,:,:]), r_neighbour)
        rand_vec = 2 .*(rand(n_particles,2).-0.5)
        for i in 1:n_particles
            # Component velocities
            if t == 1
                u_inertia = unitvec(u[t,i,:])
            else
                u_inertia = unitvec(x[t,i,:]-x[t-1,i,:])
            end
            u_cog = unitvec(indiv_best[i,:].-x[t,i,:])
            u_soc = unitvec(global_best.-x[t,i,:])
            u_neighbour = unitvec(mean(indiv_best[idxs[i],:].-repeat(transpose(x[t,i,:]),length(idxs[i])), dims = 1)[1,:])
            u_rand = unitvec(rand_vec[i,:])
            # Recalculate velocity with swarm
            u_influence = unitvec(ϕ_c*r_c[i]*u_cog .+ ϕ_s*r_s[i]*u_soc.+ ϕ_n*u_neighbour .+ ϕ_r*u_rand)
            u[t,i,:] = M.*u_inertia .+ (1-M).*u_influence
            # Calculate next position
            x[t+1,i,:] = x[t,i,:] .+ dt.*u[t,i,:]

            new_indiv_value = eval_PI_val(x[t+1,i,:], samples_TB, weights)

            # Check for new individual best
            if new_indiv_value > indiv_value[i]
                indiv_value[i] = deepcopy(new_indiv_value)
                indiv_best[i,:] = deepcopy(x[t+1,i,:])

                # Check for new global best
                if new_indiv_value > global_value
                    global_value = deepcopy(new_indiv_value)
                    global_best = deepcopy(x[t+1,i,:])
                end
            end
        end
    end
    return x
end

# Uses clustering to identify homological features from PI and returns list of centroids
function getFeatures(x; r_neighbour = 0.07, min_cluster_size = 7)
    dbscan_result = dbscan(transpose(x[end,:,:]), r_neighbour, min_cluster_size = min_cluster_size)
    centroids = zeros(length(dbscan_result),2)
    for i in 1:length(dbscan_result)
        cluster = dbscan_result[i]
        centroids[i,:] = mean(x[end,cluster.core_indices,:], dims = 1)
    end
    return centroids
end

# Extract features from a given collection of point cloud and weight data of PI
function featurise(samples_TB, weights; n_particles = 80, iterations = 120,
    dt = 0.02, r_neighbour = 0.07, min_cluster_size = 7)

    x = PSO(samples_TB,weights, n_particles = n_particles, iterations = iterations, dt = dt)

    centroids = getFeatures(x, r_neighbour = r_neighbour, min_cluster_size = min_cluster_size)
    return centroids
end

# Featurise all PI for all τ
function featuriseAll(scaled_TB, w_collection)
    centroids = Array{Any}(undef, length(scaled_TB))
    Threads.@threads for i in ProgressBar(1:length(scaled_TB), leave = false)
        centroids[i] = featurise(scaled_TB[i], w_collection[i])
    end
    return centroids
end

# Given a collection of centroids, convert into points (time, lifetime) ready for scatterplot
function featureBif(centroids; dt = 0.01)
    points = Array{Float64}(undef, (0,3))
    for i in 1:length(centroids_collection[1])
            lifetimes = deepcopy(centroids_collection[1][i])
            lifetimes = hcat(i*dt.*ones(size(lifetimes)[1]), lifetimes)
            points = vcat(points, lifetimes)
    end
    return points
end

# Calculate zcolor of scatterplot points for features based on the intensity of the PI
function calc_bif_weights(points, scaled_TB, w_collection; normalise = false)
        bif_weights = zeros(size(points)[1])

        for i in 1:length(bif_weights)
                time_idx = Int(points[i,1]÷dt)
                bif_weights[i] = eval_PI_val(points[i,2:end], scaled_TB[time_idx], w_collection[time_idx])
        end
        if normalise
            bif_weights = bif_weights./maximum(bif_weights)
        end
        return bif_weights
end


# %% Representative Cycles
function extract_cycle_index(reconstructed_cycle)
    # Extracts the index of points for the target cycle
    indexes = zeros(Int,length(reconstructed_cycle)*2)

    for i in 1:length(reconstructed_cycle)
        indexes[((i-1)*2+1):(i*2)] .= Ripserer.vertices(reconstructed_cycle[i])
    end

    indexes = sort!(indexes)[1:2:end]
    return indexes
end

# %% Analyse collection for PI information without normalisation and difference axis values
function calculate_raw_collection(T_B_collection)
        collection = deepcopy(T_B_collection)
        for i in 1:length(collection)
                for j in 1:length(collection[i])
                        # Death vs Birth
                        collection[i][j][:,2] = collection[i][j][:,2]+collection[i][j][:,1]

                        # Death/Birth vs Birth
                        # collection[i][j][:,2] = (collection[i][j][:,2]+collection[i][j][:,1])./collection[i][j][:,1]
                end
        end
        return collection
end
