module OnlineLearning
using Reexport
# package code goes here
@reexport using LossFunctions
@reexport using PenaltyFunctions
using MLLabelUtils
using MLDataUtils

include("stepsizes.jl")
include("optimizer.jl")
include("onlinelinearmodel.jl")
include("updaterules.jl")
include("fit.jl")
include("predict.jl")

export conststepsize, invstepsize
export SGDParams
export OnlineModel
export fit!
export decision_function!, decision_function, predict

end # module
