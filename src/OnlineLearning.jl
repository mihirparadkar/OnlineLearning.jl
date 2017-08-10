module OnlineLearning
using Reexport
# package code goes here
@reexport using LossFunctions
@reexport using PenaltyFunctions
@reexport using LearnBase
using MLLabelUtils
using MLDataUtils

include("misc.jl")
include("multiclassloss.jl")
include("stepsizes.jl")
include("optimizer.jl")
include("onlinelinearmodel.jl")
include("updaterules.jl")
include("fit.jl")
include("predict.jl")

export MultinomialLogitLoss, MulticlassL1HingeLoss, OVRLoss
export Optimizer, OptParams, SGDParams, NesterovParams, AdagradParams
export build_optimizer, allocate_storage, updateparams!
export OnlineModel, OnlineClassifier, OnlineRegressor, OnlineRanker, OnlineMultiClassifier
export partialfit!
export decision_func!, decision_func, predict, loss

end # module
