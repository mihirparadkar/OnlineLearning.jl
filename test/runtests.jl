using OnlineLearning
using Base.Test

# write your own tests here
Xsamp = randn(3,10)
Xsampf = randn(Float32,(3,10))
ysamp = rand(Bool, 10)

@test_nowarn OnlineLinearClassifier(L2HingeLoss(), L2Penalty(), Xsamp, rand(Bool, 10), SGDOptimizer(conststepsize(1.0)))
@test_nowarn SGDClassifier(Xsampf, ysamp)
