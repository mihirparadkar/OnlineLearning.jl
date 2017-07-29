using OnlineLearning
using Base.Test

# write your own tests here
Xsampf = randn(Float32,(10, 128));
wtrue = randn(Float32, 10)*10;
ytrue = (Xsampf'wtrue + 3randn(Float32, 128)) .> 0
yreg = (Xsampf'wtrue + 3randn(Float32, 128))

@test_nowarn OnlineModel(Xsampf, ytrue, L2HingeLoss(), L2Penalty(), SGDParams())
o = OnlineModel(Xsampf, ytrue, L2HingeLoss(), L2Penalty(), SGDParams())
ydec = predict(o, Xsampf[:,1:16])
println(sum(ytrue[1:16] .!= ydec))
fit!(o, Xsampf, ytrue, epochs=10)

or = OnlineModel(Xsampf, yreg, L2DistLoss(), L1Penalty(), SGDParams())
fit!(or, Xsampf, yreg, epochs=10)

ydec = predict(o, Xsampf[:,1:16])
println(sum(ytrue[1:16] .!= ydec))

ydec = decision_function(or, Xsampf[:,1:16])
println(zip(yreg[1:16], ydec)...)
