using OnlineLearning
using Base.Test

# write your own tests here
Xsampf = randn(Float32,(10, 128))
wtrue = randn(Float32, 10)*10;
ytrue = (Xsampf'wtrue + 3randn(Float32, 128)) .> 0
yreg = (Xsampf'wtrue + 3randn(Float32, 128))
yord = Int.(round.(clamp.(yreg, 1, 4)))

wmulti = randn(Float32, (4, 10))*10
ymulti = wmulti * Xsampf .+ 3randn(Float32, (4, 128))

@test_nowarn OnlineModel(Xsampf, ytrue, L2HingeLoss(), L2Penalty(), SGDParams())
@test_nowarn OnlineModel(Xsampf, ymulti, L1DistLoss(), L2Penalty(), SGDParams())
o = OnlineModel(Xsampf, ytrue, L2HingeLoss(), L2Penalty(), SGDParams())
ydec = predict(o, Xsampf[:,1:16])
println(sum(ytrue[1:16] .!= ydec))
fit!(o, Xsampf, ytrue, epochs=10)

or = OnlineRegressor(Xsampf, yreg)
fit!(or, Xsampf, yreg, epochs=10)

ydec = predict(o, Xsampf[:,1:16])
println(sum(ytrue[1:16] .!= ydec))

ydec = decision_func(or, Xsampf[:,1:16])
println(zip(yreg[1:16], ydec)...)

oord = OnlineModel(Xsampf, yord, OrdinalMarginLoss(L1HingeLoss(), 5),
                                    scaled(L1Penalty(), 0.1),
                                    SGDParams())
ydec = predict(oord, Xsampf[:,1:16])
println(ydec)
fit!(oord, Xsampf, yord, epochs=100, verbose=false)
ydec = predict(oord, Xsampf[:,1:16])
println(ydec)
println(yord[1:16])

omulti = OnlineModel(Xsampf, ymulti, L1DistLoss(), L2Penalty(), SGDParams())
fit!(omulti, Xsampf, ymulti, epochs=10)
