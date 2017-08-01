######################### OPTIMIZER DEFINITIONS ################################

abstract type Optimizer end

abstract type OptParams end #Holds everything except permanent storage

##########################  SGD OPTIMIZER  #####################################

mutable struct SGDOptimizer <: Optimizer
    t::Int #The current epoch of training
    newstepsize::Function
end

struct SGDParams <: OptParams
    newstepsize::Function
end

function SGDParams()
    SGDParams(conststepsize(0.01))
end

function build_optimizer(params::SGDParams, weights::Array)
    SGDOptimizer(1, params.newstepsize)
end

mutable struct SGDStorage{T <: AbstractFloat, N}
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

########################  ADAGRAD OPTIMIZER ##########################
#=
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
=#
