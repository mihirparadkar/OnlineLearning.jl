struct RegularizedLoss{L <: Loss, P <: Penalty}
    loss::L
    penalty::P
end

#Parameters for any linear model implemented here
struct Model{T <: AbstractFloat}
    weights::AbstractMatrix{T}
    bias::Vector{T}
end

function Model{T}(Xsamp::AbstractMatrix{T}, ysamp::RowVector)
    initweights = randn(T, size(Xsamp, 1))'
    initbias = zeros(T,1)
    Model(initweights, initbias)
end

function Model{T}(Xsamp::AbstractMatrix{T}, ysamp::RowVector{<:Unsigned})
    initweights = randn(T, (maximum(ysamp), size(Xsamp, 1)))
    initbias = zeros(T, maximum(ysamp))
    Model(initweights, initbias)
end

######################### Label Info ######################################
abstract type LabelInfo end

struct RealValued{T <: AbstractFloat} <: LabelInfo end

function LabelInfo{T <: AbstractFloat}(ysamp::AbstractMatrix{T})
    RealValued{T}()
end

struct BinaryValued{T <: Bool} <: LabelInfo end

function LabelInfo{T <: Bool}(ysamp::AbstractMatrix{T})
    BinaryValued{T}()
end

struct IntValued{T <: Signed} <: LabelInfo
    nlevels::T
end

function LabelInfo{T <: Signed}(ysamp::AbstractMatrix{T})
    IntValued{T}(maximum(ysamp))
end

struct CategoricalValued{T <: Unsigned} <: LabelInfo
    nlevels::T
end

function LabelInfo{T <: Unsigned}(ysamp::AbstractMatrix{T})
    CategoricalValued{T}(maximum(ysamp))
end
########################## MODEL DEFINITION ###################################
################################################################################

struct OnlineModel{D<:AbstractFloat,L<:Number,O<:Optimizer} #Type-parameterized by data-type, label-type, optimizer type
    obj::RegularizedLoss
    mod::Model{D}
    opt::O
    enc::LabelInfo
end

function OnlineModel(Xsamp::AbstractMatrix, ysamp::AbstractArray,
                    loss::Loss, penalty::Penalty,
                    optparams::OptParams)
    ysamp = correctdims(ysamp)
    obj = RegularizedLoss(loss, penalty)
    mod = Model(Xsamp, ysamp)
    opt = build_optimizer(optparams, mod.weights)
    enc = LabelInfo(ysamp)
    D = eltype(Xsamp)
    L = eltype(ysamp)
    O = typeof(opt)
    OnlineModel{D, L, O}(obj, mod, opt, enc)
end

function OnlineRegressor{L<:AbstractFloat}(Xsamp::AbstractMatrix, ysamp::AbstractArray{L};
                                        loss::Loss=HuberLoss(),
                                        penalty::Penalty=scaled(L2Penalty(),0.01),
                                        optparams::OptParams=SGDParams())
    OnlineModel(Xsamp, ysamp, loss, penalty, optparams)
end

function OnlineClassifier{L<:Bool}(Xsamp::AbstractMatrix, ysamp::AbstractArray{L};
                                loss::Loss=ModifiedHuberLoss(),
                                penalty::Penalty=scaled(L2Penalty(),0.01),
                                optparams::OptParams=SGDParams())
    OnlineModel(Xsamp, ysamp, loss, penalty, optparams)
end

function OnlineRanker{L<:Signed}(Xsamp::AbstractMatrix, ysamp::AbstractArray{L};
                                loss::Loss=OrdinalMarginLoss(ModifiedHuberLoss(), maximum(ysamp)),
                                penalty::Penalty=scaled(L2Penalty(),0.01),
                                optparams::OptParams=SGDParams())
    OnlineModel(Xsamp, ysamp, loss, penalty, optparams)
end

#=
TODO:
Enable multiclass and multilabel models
switch over weights and 1-dim labels to use RowVectors for consistency?
Nope, use matrices all the time instead, dispatching on the type
Weights use a column per feature
Bias is a column vector
=#
