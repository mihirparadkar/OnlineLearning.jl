import MLLabelUtils: LabelEncoding

#Encodings are stored so that predictions return the
#type of label the model was given
const BinaryEncoding{T <: Number} = LabelEncoding{T, 2, 1}

struct RegularizedLoss{L <: Loss, P <: Penalty}
    loss::L
    penalty::P
end

#Parameters for any linear model implemented here
mutable struct ModelParams{T <: AbstractFloat}
    weights::Vector{T}
    bias::T
end

function ModelParams{T}(Xsamp::AbstractMatrix{T})
    initweights = randn(T, size(Xsamp, 1))
    initbias = zero(T)
    ModelParams(initweights, initbias)
end

########################## MODEL DEFINITIONS ###################################
################################################################################

############################ REGRESSORS ########################################
abstract type OnlineLinearModel{T <: AbstractFloat, O <: Optimizer} end

struct OnlineLinearRegressor{T <: AbstractFloat, O <: Optimizer} <: OnlineLinearModel{T,O}
    obj::RegularizedLoss{<:DistanceLoss, <:Penalty}
    modparams::ModelParams{T}
    optparams::O
end

function OnlineLinearRegressor{T, O}(loss::DistanceLoss, penalty::Penalty,
                                    Xsamp::AbstractMatrix{T}, opt::O)
    obj = RegularizedLoss(loss, penalty)
    mod = ModelParams(Xsamp)
    OnlineLinearRegressor(obj, mod, opt)
end

function decodelabels{T <: AbstractFloat}(mod::OnlineLinearRegressor{T, <: Optimizer}, y::Vector)
    convert(Vector{T}, y)
end

function encodelabels{T <: AbstractFloat}(mod::OnlineLinearRegressor{T, <: Optimizer}, output::Vector{T})
    output
end

## SGD REGRESSOR ##

function SGDRegressor{T}(Xsamp::AbstractMatrix{T};
                        loss::DistanceLoss=L2DistLoss(),
                        penalty::Penalty=scaled(L2Penalty(), 0.01),
                        newstepsize::Function=conststepsize(0.01))
    opt = SGDOptimizer(newstepsize)
    OnlineLinearRegressor(loss, penalty, Xsamp, opt)
end

## ADAGRAD REGRESSOR ##

function AdaGradRegressor{T}(Xsamp::AbstractMatrix{T};
                            loss::DistanceLoss=L2DistLoss(),
                            penalty::Penalty=scaled(L2Penalty(), 0.01),
                            newstepsize::Function=conststepsize(0.01))
    opt = AdaGradOptimizer(Xsamp, newstepsize)
    OnlineLinearRegressor(loss, penalty, Xsamp, opt)
end

######################## CLASSIFIERS ###########################################

struct OnlineLinearClassifier{T <: AbstractFloat, O <: Optimizer} <: OnlineLinearModel{T,O}
    obj::RegularizedLoss{<:MarginLoss, <:Penalty}
    modparams::ModelParams{T}
    optparams::O
    encoding::BinaryEncoding{<:Number}
end

function OnlineLinearClassifier{T,O}(loss::MarginLoss, penalty::Penalty,
                                    Xsamp::AbstractMatrix{T}, ysamp::Vector,
                                    opt::O)
    obj = RegularizedLoss(loss, penalty)
    mod = ModelParams(Xsamp)
    encoding = labelenc(ysamp)
    if !(encoding isa BinaryEncoding)
        error("Labels must only have two possible values, but given labels have more")
    end
    OnlineLinearClassifier(obj, mod, opt, encoding)
end

function decodelabels{T <: AbstractFloat}(olc::OnlineLinearClassifier{T, <:Optimizer}, y::Vector)
    convertlabel(LabelEnc.MarginBased{T}, y, olc.encoding)
end

## SGD CLASSIFIER ##

function SGDClassifier{T}(Xsamp::AbstractMatrix{T}, ysamp::Vector;
                        loss::MarginLoss=HingeLoss(),
                        penalty::Penalty=scaled(L2Penalty(), 0.01),
                        newstepsize::Function=conststepsize(0.01))
    opt = SGDOptimizer(newstepsize)
    OnlineLinearClassifier(loss, penalty, Xsamp, ysamp, opt)
end

## ADAGRAD CLASSIFIER ##

function AdaGradClassifier{T}(Xsamp::AbstractMatrix{T}, ysamp::Vector;
                            loss::MarginLoss=L2HingeLoss(),
                            penalty::Penalty=scaled(L2Penalty(), 0.01),
                            newstepsize::Function=conststepsize(0.01))
    opt = AdaGradOptimizer(Xsamp, newstepsize)
    OnlineLinearClassifier(loss, penalty, Xsamp, ysamp, opt)
end
