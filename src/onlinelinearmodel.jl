struct RegularizedLoss{L <: Loss, P <: Penalty}
    loss::L
    penalty::P
end

#Parameters for any linear model implemented here
struct Model{T <: AbstractFloat}
    weights::Vector{T}
    bias::Vector{T}
end

function Model{T}(Xsamp::AbstractMatrix{T}, ysamp::DenseVector)
    initweights = randn(T, size(Xsamp, 1))
    initbias = zeros(T,1)
    Model(initweights, initbias)
end

########################## MODEL DEFINITION ###################################
################################################################################

struct OnlineModel{D<:AbstractFloat,L<:Number,O<:Optimizer} #Type-parameterized by data-type, label-type, optimizer type
    obj::RegularizedLoss
    mod::Model{D}
    opt::O
end

const OnlineClassifier{D<:AbstractFloat,L<:Bool,O<:Optimizer} = OnlineModel{D,L,O}

const OnlineRegressor{D<:AbstractFloat,L<:AbstractFloat,O<:Optimizer} = OnlineModel{D,L,O}

const OnlineRanker{D<:AbstractFloat,L<:Integer,O<:Optimizer} = OnlineModel{D,L,O}

function OnlineModel(Xsamp::AbstractMatrix, ysamp::DenseVector,
                    loss::Loss, penalty::Penalty,
                    optparams::OptParams)

    obj = RegularizedLoss(loss, penalty)
    mod = Model(Xsamp, ysamp)
    opt = build_optimizer(optparams, mod.weights)
    D = eltype(Xsamp)
    L = eltype(ysamp)
    O = typeof(opt)
    OnlineModel{D, L, O}(obj, mod, opt)
end
