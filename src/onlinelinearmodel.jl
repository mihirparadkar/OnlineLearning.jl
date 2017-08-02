struct RegularizedLoss{L <: Loss, P <: Penalty}
    loss::L
    penalty::P
end

#Parameters for any linear model implemented here
struct Model{T <: AbstractFloat, N}
    weights::Array{T, N}
    bias::Vector{T}
end

function build_model{T}(Xsamp::AbstractMatrix{T}, ysamp::DenseArray{<:Number, 1})
    initweights = randn(T, size(Xsamp, 1))
    initbias = zeros(T,1)
    Model{T, 1}(initweights, initbias)
end

function build_model{T}(Xsamp::AbstractMatrix{T}, ysamp::DenseArray{<:Number, 2})
    initweights = randn(T, (size(ysamp, 1), size(Xsamp, 1)))
    initbias = zeros(T,size(ysamp, 1))
    Model{T, 2}(initweights, initbias)
end

function build_model{T}(Xsamp::AbstractMatrix{T}, ysamp::DenseVector{<:Unsigned})
    initweights = randn(T, (Int(maximum(ysamp)), size(Xsamp, 1)))
    initbias = zeros(T, Int(maximum(ysamp)))
    Model{T, 2}(initweights, initbias)
end

######################### Label Info ######################################
abstract type LabelInfo end

struct RealValued{T <: AbstractFloat} <: LabelInfo end

function LabelInfo{T <: AbstractFloat}(ysamp::DenseArray{T})
    RealValued{T}()
end

struct BinaryValued{T <: Bool} <: LabelInfo end

function LabelInfo{T <: Bool}(ysamp::DenseArray{T})
    BinaryValued{T}()
end

struct IntValued{T <: Signed} <: LabelInfo
    nlevels::T
end

function LabelInfo{T <: Signed}(ysamp::AbstractArray{T})
    IntValued{T}(maximum(ysamp))
end

struct CategoricalValued{T <: Unsigned} <: LabelInfo
    nlevels::T
end

function LabelInfo{T <: Unsigned}(ysamp::DenseVector{T})
    CategoricalValued{T}(maximum(ysamp))
end
########################## MODEL DEFINITION ###################################
################################################################################

struct OnlineModel{D<:AbstractFloat,L<:Number,O<:Optimizer,N} #Type-parameterized by data-type, label-type, optimizer type
    obj::RegularizedLoss
    mod::Model{D, N}
    opt::O
    enc::LabelInfo
end

function OnlineModel{N}(Xsamp::AbstractMatrix, ysamp::DenseArray{<:Number, N},
                    loss::Loss, penalty::Penalty,
                    optparams::OptParams)
    obj = RegularizedLoss(loss, penalty)
    mod = build_model(Xsamp, ysamp)
    opt = build_optimizer(optparams, mod.weights)
    enc = LabelInfo(ysamp)
    D = eltype(Xsamp)
    L = eltype(ysamp)
    O = typeof(opt)
    OnlineModel{D, L, O, N}(obj, mod, opt, enc)
end

"""
Constructor for online multiclass classifiers, where labels are unsigned ints
to distinguish them from ordinal labels
"""
function OnlineModel(Xsamp::AbstractMatrix, ysamp::DenseVector{<:Unsigned},
                    loss::Loss, penalty::Penalty,
                    optparams::OptParams)
    obj = RegularizedLoss(loss, penalty)
    mod = build_model(Xsamp, ysamp)
    opt = build_optimizer(optparams, mod.weights)
    enc = LabelInfo(ysamp)
    D = eltype(Xsamp)
    L = eltype(ysamp)
    O = typeof(opt)
    OnlineModel{D, L, O, 2}(obj, mod, opt, enc)
end

function OnlineRegressor{L<:AbstractFloat}(Xsamp::AbstractMatrix, ysamp::DenseArray{L};
                                        loss::DistanceLoss=HuberLoss(),
                                        penalty::Penalty=scaled(L2Penalty(),0.01),
                                        optparams::OptParams=SGDParams())
    OnlineModel(Xsamp, ysamp, loss, penalty, optparams)
end

function OnlineClassifier{L<:Bool}(Xsamp::AbstractMatrix, ysamp::DenseArray{L};
                                loss::MarginLoss=ModifiedHuberLoss(),
                                penalty::Penalty=scaled(L2Penalty(),0.01),
                                optparams::OptParams=SGDParams())
    OnlineModel(Xsamp, ysamp, loss, penalty, optparams)
end

function OnlineRanker{L<:Signed}(Xsamp::AbstractMatrix, ysamp::DenseArray{L};
                                loss::Loss=OrdinalMarginLoss(ModifiedHuberLoss(), maximum(ysamp)),
                                penalty::Penalty=scaled(L2Penalty(),0.01),
                                optparams::OptParams=SGDParams())
    OnlineModel(Xsamp, ysamp, loss, penalty, optparams)
end

function OnlineMultiClassifier{L<:Unsigned}(Xsamp::AbstractMatrix, ysamp::DenseArray{L};
                                loss::CategoricalLoss=MulticlassL1HingeLoss(maximum(ysamp)),
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
