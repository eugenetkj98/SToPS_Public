using LinearAlgebra
using StatsBase
using Random
using ProgressBars
using Statistics

"""
This file contains a collection of chaotic dynamics systems, integrators,
and delay embedding methods for data generation and analaysis.
"""
# %% Helper functions

function lorenz!(X)
    sigma = 10
    beta = 8/3
    rho = 28

    x, y, z = X

    dx = sigma*(y-x)
    dy = x*(rho-z)-y
    dz = x*y-beta*z

    output = [dx, dy, dz]
    return output
end

function periodic_lorenz!(X)
    sigma = 10
    beta = 8/3
    rho = 160

    x, y, z = X

    dx = sigma*(y-x)
    dy = x*(rho-z)-y
    dz = x*y-beta*z

    output = [dx, dy, dz]
    return output
end

function rossler!(X)
    a = 0.2
    b = 0.2
    c = 5.7

    x, y, z = X

    dx = -y-z
    dy = x+a*y
    dz = b + z*(x-c)

    output = [dx, dy, dz]
    return output
end

# Phase coherent Rossler
function rossler_PC!(X)
    a = 0.165
    b = 0.4
    c = 8.5

    x, y, z = X

    dx = -y-z
    dy = x+a*y
    dz = b + z*(x-c)

    output = [dx, dy, dz]
    return output
end

# Non-phase coherent rossler
function rossler_NPC!(X)
    a = 0.265
    b = 0.4
    c = 8.5

    x, y, z = X

    dx = -y-z
    dy = x+a*y
    dz = b + z*(x-c)

    output = [dx, dy, dz]
    return output
end

function chua!(X)
    #Parameters
    k = -1
    β = 53.612186
    γ = -0.75087096
    α = 17

    #Cubic nonlinearity
    a = -0.0375582129
    b = -0.8415410391
    f(x) = a*x^3 + b*x

    x,y,z = X
    dx = k*(y-x+z)
    dy = k*α*(x-y-f(y))
    dz = k*(-β*x-γ*z)

    output = [dx,dy,dz]
end

function halvorsen!(X)
    #Parameters Xiaogen Yin, YJ Cao, Sychronisation of Chua’s oscillator via the state observer techniqu
    a = 1.27
    b = 4

    x,y,z = X
    dx = -a*x-b*(y+z)-y^2
    dy = -a*y-b*(z+x)-z^2
    dz = -a*z-b*(x+y)-x^2

    output = [dx,dy,dz]
end

function duffing!(X)
    #Chaotic Duffing oscillator (Guckenheimer, Kanamaru)
    α = 1
    β = -1
    δ = 0.2
    γ = 0.3
    ω = 1

    x,xdot,psi = X
    dx = xdot
    dy = -δ*xdot-β*x-α*x^3+γ*cos(psi)
    dz = ω

    output = [dx,dy,dz]
end

function pendulum!(X)
    #Forced pendulum intermittency (Grebogi 1987)
    Ω = 1
    ω = 1
    v = 0.22
    p = 2.8

    phi, phidot, psi = X
    dx = phidot
    dy = -v*phidot-(Ω^2)*sin(phi)+p*cos(psi)
    dz = ω

    output = [dx,dy,dz]
end

function FHN_neuron2!(X; f = 0.12643)
    # f = 0.12643
    α = 0.1 #input Driving amplitude
    b1 = 10
    b2 = 1

    v, w, θ = X
    ω = 2*pi*f
    dv = v*(v-1)*(1-b1*v) - w + (α/ω)*cos(θ) #Sodium positive depolarisation channel
    dw = b2*v
    dθ = ω
    # dI = dI
    # ddI = -(ω^2)*I

    # output = [dv, dw, dI, ddI]
    output = [dv, dw, dθ]


    return output
end

# %% RK4 Integration

function linearise(X; model = nothing)

    if isnothing(model)
        output = lorenz!(X)
    else
        output = model(X)
    end
    return output
end


function integrate(model;dt=0.02, T=5000, RK = false, supersample = 1, dims = 3, init = nothing, noise = nothing)
    # Generate Empty Matrix
    S = zeros((T*supersample, dims))
    dS = zeros(dims)
    δt = (dt/supersample)

    # Initialise starting values
    if isnothing(init)
        S[1, :] = 1 .*(rand(Float64,size(S[1, :])).-0.5)
    else
        S[1, :] = init
    end

    for t in 1:T*supersample
        if !RK
            dS[:] = linearise(S[t, :], model = model)
        else
            # Runge Kutta Integration
            k1 = linearise(S[t, :], model = model)
            k2 = linearise(S[t, :].+δt*(k1/2), model = model)
            k3 = linearise(S[t, :].+δt*(k2/2), model = model)
            k4 = linearise(S[t, :].+δt*k3, model = model)
            dS[:] = (k1+2*(k2+k3)+k4)/6
        end

        if !isnothing(noise)
            dS = dS + noise.*randn(size(dS))
        end # Apply coupling forces

        if t < T*supersample
            S[t+1, :] = S[t, :]+δt*dS
        end
    end

    return S[1:supersample:(T*supersample),:]
end

function MackeyGlass(T, dt, lag)
    γ = 1
    β = 2
    n = 9.65
    τ = Int(lag/dt)
    x0 = 0.5
    x = Array{Float64}(undef, T+wash)
    x[1:(τ+1)] .= x0
    for i in (τ+1):(length(x)-1)
        dx = β*((x[i-τ])/(1+x[i-τ]^n))-γ*x[i]
        x[i+1] = x[i] + dt*dx
    end
    return x
end

# %% Time Delay Embedding
function custom_embed(x, τ, m)
    S = zeros(length(x)-(m-1)*τ, m)

    for dim in 1:m
        S[:,dim] = x[1+(m-dim)*τ:end-(dim-1)*τ]
    end

    return S
end

function nonunif_embed(x, τ_array)
    # Sort lags in ascending order
    # sort!(τ_array)

    # Create storage array for result
    m = length(τ_array)+1
    # S = zeros(length(x)-τ_array[end], m)
    S = zeros(length(x)-sum(τ_array), m)

    # Non-uniform embedding
    for dim in 1:m
        if dim == 1
            # S[:,dim] = x[1:end-τ_array[end]]
            S[:,dim] = x[sum(τ_array)+1:end]
        elseif dim < m
            # S[:,dim] = x[(τ_array[dim-1]+1):end-τ_array[end]+τ_array[dim-1]]
            S[:,dim] = x[(sum(τ_array)+1-sum(τ_array[1:(dim-1)])):end-sum(τ_array[1:(dim-1)])]
        else
            # S[:,dim] = x[(τ_array[dim-1]+1):end]
            S[:,dim] = x[1:end-sum(τ_array)]
        end
    end

    return S
end

# Same as nonunif_embed except tau lags are calculated relative to the first component, and not successive
function nonunif_embed2(x, τ_array)
    # Sort lags in ascending order
    sort!(τ_array)

    # Create storage array for result
    m = length(τ_array)+1
    S = zeros(length(x)-τ_array[end], m)

    # Non-uniform embedding
    for dim in 1:m
        if dim == 1
            S[:,dim] = x[1:end-τ_array[end]]
        elseif dim < m
            S[:,dim] = x[(τ_array[dim-1]+1):end-τ_array[end]+τ_array[dim-1]]
        else
            S[:,dim] = x[(τ_array[dim-1]+1):end]
        end
    end

    return S
end
