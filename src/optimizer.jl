######################### OPTIMIZER DEFINITIONS ################################

abstract type Optimizer end

abstract type OptParams end #Holds everything except permanent storage

##########################  SGD/MOMENTUM OPTIMIZER  ############################

mutable struct SGDOptimizer{T<:AbstractFloat, N} <: Optimizer
    t::Int #The current epoch of training
    prevgrad::Array{T, N}
    prevgradbias::Vector{T}
    η0::T #Initial stepsize
    decay::T #Multiplied by t to get new stepsize
    power_t::T #t is raised to this power in stepsize formula
    momentum::T #proportion of gradient update from previous updates
end

"""
Stochastic Gradient Descent parameters
The parameter update rule is given by
η₀ / (1 + decay * t ^ power_t), where t is the current epoch of training

The factor of momentum is given by `momentum`, where 0.0 means standard SGD
This should be a number between 0.0 and 1.0
"""
struct SGDParams <: OptParams
    η0::Float64 #Initial stepsize
    decay::Float64 #Multiplied by t to get new stepsize (bigger shrinks faster)
    power_t::Float64 #t is raised to this power in stepsize formula (bigger shrinks faster)
    momentum::Float64 #proportion of gradient update from previous updates
end

function SGDParams(;η0=0.05, decay=0.0, power_t=0.25, momentum=0.0)
    SGDParams(η0, decay, power_t, momentum)
end

function build_optimizer{T <: AbstractFloat}(params::SGDParams, weights::Vector{T})
    SGDOptimizer{T, 1}(0, zeros(weights), zeros(T, 1), params.η0, params.decay, params.power_t, params.momentum)
end

function build_optimizer{T <: AbstractFloat}(params::SGDParams, weights::Matrix{T})
    SGDOptimizer{T, 2}(0, zeros(weights), zeros(T, size(weights, 1)),
                    params.η0, params.decay, params.power_t, params.momentum)
end

struct SGDStorage{T <: AbstractFloat, N}
    grad::Array{T, N}
    derv::Array{T, N}
    gradbias::Vector{T}
end

function allocate_storage{T <: AbstractFloat}(weights::Vector{T}, batchlen::Int, opt::SGDOptimizer)
    nfeats = size(weights)
    SGDStorage(zeros(T, nfeats), zeros(T, batchlen), zeros(T, 1))
end

function allocate_storage{T <: AbstractFloat}(weights::Matrix{T}, batchlen::Int, opt::SGDOptimizer)
    nfeats = size(weights, 2)
    nlabels = size(weights, 1)
    SGDStorage(zeros(T, (nlabels, nfeats)), zeros(T, (nlabels, batchlen)), zeros(T, nlabels))
end

########################  NESTEROV OPTIMIZER ##################################

mutable struct NesterovOptimizer{T<:AbstractFloat, N} <: Optimizer
    t::Int #The current epoch of training
    prevgrad::Array{T, N}
    prevgradbias::Vector{T}
    η0::T #Initial stepsize
    decay::T #Multiplied by t to get new stepsize
    power_t::T #t is raised to this power in stepsize formula
    momentum::T #proportion of gradient update from previous updates
end

"""
Nesterov Update parameters
The parameter update rule is given by
η₀ / (1 + decay * t ^ power_t), where t is the current epoch of training

The factor of momentum is given by `momentum`, where 0.0 means no momentum
This should be a number between 0.0 and 1.0
"""
struct NesterovParams <: OptParams
    η0::Float64 #Initial stepsize
    decay::Float64 #Multiplied by t to get new stepsize (bigger shrinks faster)
    power_t::Float64 #t is raised to this power in stepsize formula (bigger shrinks faster)
    momentum::Float64 #proportion of gradient update from previous updates
end

function NesterovParams(;η0=0.05, decay=0.0, power_t=0.25, momentum=0.0)
    NesterovParams(η0, decay, power_t, momentum)
end

function build_optimizer{T <: AbstractFloat}(params::NesterovParams, weights::Vector{T})
    NesterovOptimizer{T, 1}(0, zeros(weights), zeros(T, 1), params.η0, params.decay, params.power_t, params.momentum)
end

function build_optimizer{T <: AbstractFloat}(params::NesterovParams, weights::Matrix{T})
    NesterovOptimizer{T, 2}(0, zeros(weights), zeros(T, size(weights, 1)),
                    params.η0, params.decay, params.power_t, params.momentum)
end

struct NesterovStorage{T <: AbstractFloat, N}
    grad::Array{T, N}
    derv::Array{T, N}
    gradbias::Vector{T}
    nextweights::Array{T, N}
    nextbias::Vector{T}
end

function allocate_storage{T <: AbstractFloat}(weights::Vector{T}, batchlen::Int, opt::NesterovOptimizer)
    nfeats = size(weights)
    NesterovStorage(zeros(T, nfeats), zeros(T, batchlen), zeros(T, 1),
                    zeros(T, nfeats), zeros(T, 1))
end

function allocate_storage{T <: AbstractFloat}(weights::Matrix{T}, batchlen::Int, opt::NesterovOptimizer)
    nfeats = size(weights, 2)
    nlabels = size(weights, 1)
    NesterovStorage(zeros(T, (nlabels, nfeats)), zeros(T, (nlabels, batchlen)), zeros(T, nlabels),
                    zeros(T, (nlabels, nfeats)), zeros(T, nlabels))
end


########################  ADAGRAD OPTIMIZER ##########################

mutable struct AdagradOptimizer{T <: AbstractFloat, N} <: Optimizer
    t::Int
    sqgrads::Array{T, N}
    sqbiasgrads::Vector{T}
    η0::T #Initial stepsize
    decay::T #Multiplied by t to get new stepsize
    power_t::T #t is raised to this power in stepsize formula
    ϵ::T #Amount of noise to avoid division by zero
end

"""
AdagradParams <: OptParams

The parameter update rule is given by function
η₀ / (1 + decay * t ^ power_t), where t is the current epoch of training

ϵ is an amount of noise, usually around 1e-8, to prevent division by zero
"""
struct AdagradParams <: OptParams
    η0::Float64
    decay::Float64
    power_t::Float64
    ϵ::Float64
end

function AdagradParams(;η0=0.1, decay=0.0, power_t=0.25, ϵ=1e-8)
    AdagradParams(η0, decay, power_t, ϵ)
end

function build_optimizer{T <: AbstractFloat}(params::AdagradParams, weights::Vector{T})
    AdagradOptimizer{T, 1}(0, zeros(weights), zeros(T, 1), params.η0, params.decay, params.power_t, params.ϵ)
end

function build_optimizer{T <: AbstractFloat}(params::AdagradParams, weights::Matrix{T})
    AdagradOptimizer{T, 2}(0, zeros(weights), zeros(T, size(weights, 1)),
                            params.η0, params.decay, params.power_t, params.ϵ)
end

struct AdagradStorage{T <: AbstractFloat, N}
    grad::Array{T, N}
    derv::Array{T, N}
    gradbias::Vector{T}
end

function allocate_storage{T <: AbstractFloat}(weights::Vector{T}, batchlen::Int, opt::AdagradOptimizer)
    nfeats = size(weights)
    AdagradStorage(zeros(T, nfeats), zeros(T, batchlen), zeros(T, 1))
end

function allocate_storage{T <: AbstractFloat}(weights::Matrix{T}, batchlen::Int, opt::AdagradOptimizer)
    nfeats = size(weights, 2)
    nlabels = size(weights, 1)
    AdagradStorage(zeros(T, (nlabels, nfeats)), zeros(T, (nlabels, batchlen)), zeros(T, nlabels))
end
