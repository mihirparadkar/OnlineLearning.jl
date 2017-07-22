######################### OPTIMIZER DEFINITIONS ################################

abstract type Optimizer end

##########################  SGD OPTIMIZER  #####################################

mutable struct SGDOptimizer <: Optimizer
    t::Int #The current epoch of training
    newstepsize::Function
end

function SGDOptimizer(newstepsize::Function)
    SGDOptimizer(1, newstepsize)
end

mutable struct SGDStorage{T <: AbstractFloat}
    grad::Vector{T}
    derv::Vector{T}
    gradbias::T
end

function allocate_storage{T <: AbstractFloat}(Xsamp::AbstractMatrix{T}, batchlen::Int, opt::SGDOptimizer)
    nfeats = size(Xsamp, 1)
    SGDStorage(zeros(T, nfeats), zeros(T, batchlen), zero(T))
end

########################  ADAGRAD OPTIMIZER ##########################

mutable struct AdaGradOptimizer{T <: AbstractFloat} <: Optimizer
    t::Int
    newstepsize::Function
    Binvsq::Vector{T} #The sum of squares of each component of the gradient
    avgsqbias::T #Sum of squared biases
end

function AdaGradOptimizer{T <: AbstractFloat}(Xsamp::AbstractMatrix{T}, newstepsize::Function = conststepsize(1.))
    Binvsq = zeros(T, size(Xsamp, 1))
    t = 1
    AdaGradOptimizer(t, newstepsize, Binvsq, one(T))
end

mutable struct AdaGradStorage{T <: AbstractFloat}
    grad::Vector{T}
    derv::Vector{T}
    gradbias::T
end

function allocate_storage{T <: AbstractFloat}(Xsamp::AbstractMatrix{T}, batchlen::Int, opt::AdaGradOptimizer{T})
    nfeats = size(Xsamp, 1)
    AdaGradStorage(zeros(T, nfeats), zeros(T, batchlen), zero(T))
end

####################### MOMENTUM OPTIMIZER ###########################
