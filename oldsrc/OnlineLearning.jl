module OnlineLearning
using Reexport
# package code goes here
@reexport using LossFunctions
@reexport using PenaltyFunctions
using MLLabelUtils
using MLDataUtils

include("stepsizes.jl")
include("optimizers.jl")
include("onlinelinearmodel.jl")
include("updaterules.jl")
include("fit.jl")


export SGDRegressor, AdaGradRegressor
export SGDClassifier, AdaGradClassifier
export conststepsize, invstepsize
export fit!

end # module
