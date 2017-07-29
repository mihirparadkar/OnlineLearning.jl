struct RegularizedLoss{L <: Loss, P <: Penalty}
    loss::L
    penalty::P
end

#Parameters for any linear model implemented here
struct ModelParams{T <: AbstractFloat}
    weights::VecOrMat{T}
    bias::Vector{T}
end

function Model{T}(Xsamp::AbstractMatrix{T}, ysamp::Vector)
    initweights = randn(T, size(Xsamp, 1))
    initbias = zeros(T,1)
    ModelParams(initweights, initbias)
end

function Model{T}(Xsamp::AbstractMatrix{T}, ysamp::Matrix)
    initweights = randn(T, size(Xsamp, 1), size(ysamp, 1))
    initbias = zeros(T, size(ysamp, 1))
    ModelParams(initweights, initbias)
end

########################## MODEL DEFINITIONS ###################################
################################################################################

struct OnlineLinearModel{D<:AbstractFloat,L<:Number,O<:Optimizer}
    obj::RegularizedLoss
    mod::Model{D}
    opt::Optimizer
end
