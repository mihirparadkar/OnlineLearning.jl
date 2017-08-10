# OnlineLearning

[![Build Status](https://travis-ci.org/mihirparadkar/OnlineLearning.jl.svg?branch=master)](https://travis-ci.org/mihirparadkar/OnlineLearning.jl)

[![Coverage Status](https://coveralls.io/repos/mihirparadkar/OnlineLearning.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/mihirparadkar/OnlineLearning.jl?branch=master)

[![codecov.io](http://codecov.io/github/mihirparadkar/OnlineLearning.jl/coverage.svg?branch=master)](http://codecov.io/github/mihirparadkar/OnlineLearning.jl?branch=master)

A package for building, fitting, and predicting linear machine learning models on streaming data.


## A Basic Example
This package is intended for supervised machine learning on streaming data - that is,
data that doesn't necessarily all fit in main memory at once. This data might be from
a network or on disk, for example, with small amounts being loaded into memory at a time.
This package implements supervised models that can learn partially, or incrementally, from
smaller subsets of full training data.

These models support boolean labels (classification), multi-class classification,
ranking, and regression through a unified interface. At a minimum, all that are required to
build and fit a model are training data and labels, and the package will assign reasonable
default parameters.

Here is a very simple example with artificial data
```julia
X = randn(10, 800) # Training data with 10 features and 800 examples
Xval = randn(10, 200) # Validation data
wtrue = 3randn(10) # The true parameters
y = X'wtrue .+ randn(800) # Data times true parameter plus noise
yval = Xval'wtrue .+ randn(200) #Validation labels

#Construct the model and fit it over 50 epochs
regr = OnlineModel(X, y)
partialfit!(regr, X, y, epochs=50)

#Predict the output of the model on the validation data
ypred = predict(regr, Xval)
```

One thing to notice is that each *column* of the training data X represents a
sample. This is different from libraries like scikit-learn, for example. This difference
is because Julia matrices are column-major, so sampling columns from a matrix is
much more computationally efficient than sampling rows. The orientation of the labels
in multi-label classification should be the same, that is, the set of labels for one sample
should correspond to a *column* of the label matrix.

## A More Involved Example
The basic example doesn't exactly show OnlineLearning being used in a streaming context.
The data is available in its entirety, so it seems redundant to have to specify X and y
both when constructing the model and when fitting it.

This next example shows how an OnlineModel can be fit continuously over changing
input data, this time with a multi-class classification problem.
```julia
wtrue = 3randn(5, 10) # Use these parameters to generate fake data
Xsamp = zeros(10, 1) # The content of the data doesn't matter as long as the number of columns is correct
ysamp = UInt[5] # Unsigned integers are assumed to be multi-class classification, while signed integers are ordinal
                # With ordinal and multiclass data, make sure that the maximum of the sample data is the number of classes

svm = OnlineModel(Xsamp, ysamp) #Defaults to a multi-class hinge loss, so a form of SVM
ch = Float64[] #Record of validation error
for s in 1:100
    # Fake data generation to simulate a stream or other source of changing data
    X = randn(10, 200)
    Xw = wtrue*X
    y = UInt[indmax(Xw[:,i]) for i in 1:size(Xw, 2)]

    # Fit the model at 1 epochs per fake data stream, holding off the last 20% for validation
    # Also record the validation error
    partialfit!(svm, X[:,1:160], y[1:160])
    valloss = loss(svm, y[161:end], decision_func(svm, X[:,161:end]))
    push!(ch, valloss)

    #Shrink the stepsize if the most recent validation error is larger than the next-most recent
    if s >= 5 && (ch[end] - ch[end - 1] >= 0)
      svm.opt.η0 *= 0.7
    end
end
```
This example illustrates the "Online" in "OnlineLearning"; the model fitting code
sits inside the user code for data streaming and processing. This allows maximal
flexibility in controlling how the model fits. For example, additional code can be added
to monitor validation error and adjust the stepsize of the model during fitting.

## Components of an OnlineModel

At the core of an OnlineModel are three main things: an objective to minimize
(loss + regularizer), a set of parameters (weights and bias), and an optimizer
with information about stepsize, current epoch, and things like a momentum vector
or other optimizer parameters that are built up over time.

## Supported Label Data-Types

OnlineLearning supports enables single and multi-label classification, regression,
and ranking, as well as single-label multi-class classification. The problems are
indicated by the element types of the labels passed to the model. This avoids label
ambiguities, such as whether the negative class in classification is given by 0 or -1.
The mapping from problem type to label type is given below:

`<: AbstractFloat`(Float32, Float64, etc.) -> Regression

`Bool` (Array{Bool}, BitArray, etc.) -> Binary Classification

`<: Signed` (Int64, Int32, Int8, etc.) -> Ordinal Ranking
Here, the labels are expected to be between 1 and the number of ordinal levels,
so the maximum element of the `y` passed to the constructor should be the number of levels.

`<: Unsigned` (UInt64, UInt32, UInt8, etc.) -> Multiclass Classification
These should also have values between 1 and `N` where `N` is the number of classes
Similarly to the ordinal regression case, the `y` passed into the constructor should
have maximum value equal to the number of classes.

## Objective Functions

The objective function to minimize consists of a loss function, a way to measure the
error between predictions and actual values, and a regularization penalty, or a
penalty on the model's weights themselves. For regression, classification, and ranking,
the loss functions from LossFunctions.jl ( https://github.com/JuliaML/LossFunctions.jl ),
are supported. `DistanceLoss`es are supported for regression, `MarginLoss`es are
used in binary classification, and either `DistanceLoss`es or `OrdinalMarginLoss`es
are recommended for ranking.
For multi-class classification, three types of loss are implemented here.

The MultinomialLogitLoss generalizes the logistic loss function to multiple classes and
is grounded in a probabilistic interpretation.

The MulticlassL1HingeLoss generalizes the hinge loss used in SVMs to multi-class problems

OVRLoss treats the multi-class problem as separate binary classification problems and
uses an internal MarginLoss (chosen at construction) to optimize these problems.

The supported penalties are further documented in https://github.com/JuliaML/PenaltyFunctions.jl
Any (subgradient) differentiable penalty function is usable here.

## Optimizers

Multiple optimizers are supported for experimentation and because of their properties
on different problems. Currently supported optimizers are SGD (Stochastic Gradient Descent)
with or without momentum, Nesterov SGD, and Adagrad, which maintains a running sum of previous
squared gradients.

For SGDParams, NesterovParams, or AdagradParams, more information can be found by typing, for example,
`?SGDParams`.
