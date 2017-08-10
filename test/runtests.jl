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
ymultiord = Int.(round.(clamp.(ymulti, 1, 4)))
ycat = UInt32[indmax(ymulti[:,i]) for i in 1:size(ymulti, 2)]

@test_nowarn OnlineModel(Xsampf, ytrue, L2HingeLoss(), L2Penalty(), SGDParams())
@test_nowarn OnlineModel(Xsampf, ymulti, L1DistLoss(), L2Penalty(), NesterovParams())
o = OnlineModel(Xsampf, ytrue, L2HingeLoss(), L2Penalty(), NesterovParams())
ydec = predict(o, Xsampf[:,1:16])
println(sum(ytrue[1:16] .!= ydec))
partialfit!(o, Xsampf, ytrue, epochs=10)

or = OnlineRegressor(Xsampf, yreg)
partialfit!(or, Xsampf, yreg, epochs=10)

ydec = predict(o, Xsampf[:,1:16])
println(sum(ytrue[1:16] .!= ydec))

ydec = decision_func(or, Xsampf[:,1:16])
println(zip(yreg[1:16], ydec)...)

oord = OnlineModel(Xsampf, yord, OrdinalMarginLoss(L1HingeLoss(), 5),
                                    scaled(L1Penalty(), 0.1),
                                    AdagradParams())
ydec = predict(oord, Xsampf[:,1:16])
println(ydec)
partialfit!(oord, Xsampf, yord, epochs=100, verbose=false)
ydec = predict(oord, Xsampf[:,1:16])
println(ydec)
println(yord[1:16])

omulti = OnlineModel(Xsampf, ymulti, L1DistLoss(), L2Penalty(), SGDParams())
partialfit!(omulti, Xsampf, ymulti, epochs=10)

omultiord = OnlineRanker(Xsampf, ymultiord)
ydec = predict(omultiord, Xsampf[:,1:6])
println(ydec)
partialfit!(omultiord, Xsampf, ymultiord, epochs=100, verbose=false)
ydec = predict(omultiord, Xsampf[:,1:6])
println(ydec)
println(ymultiord[:,1:6])

@test_nowarn OnlineMultiClassifier(Xsampf, ycat)
ocat = OnlineMultiClassifier(Xsampf, ycat, loss=MultinomialLogitLoss(4))
ypred = predict(ocat, Xsampf[:,1:6])
println(ypred)
partialfit!(ocat, Xsampf, ycat, epochs=100, verbose=false)
ypred = predict(ocat, Xsampf[:,1:6])
println(ypred)
println(ycat[1:6])
