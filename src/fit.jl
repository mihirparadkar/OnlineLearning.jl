function defaultbatchsize(numobs::Int)
    #The nearest power of 2 above the square root of the number of observations
    numobs |> sqrt |> log2 |> ceil |> exp2 |> Int
end

function fit!{T <: AbstractFloat}(mod::OnlineLinearModel,
                                X::AbstractMatrix{T}, y::Vector;
                                shuffle::Bool=true,
                                batchsize::Int=defaultbatchsize(size(X,2)),
                                epochs::Int=1,
                                verbose::Bool=true)

    yencoded = decodelabels(mod, y)
    nfeats = size(X, 1)
    if shuffle
        Xbatch, ybatch = map(Array, shuffleobs((X, yencoded)))
    else
        Xbatch, ybatch = X, yencoded
    end

    storage = allocate_storage(Xbatch, batchsize, mod.optparams)

    for iter in 1:epochs
        for (Xmini, ymini) in eachbatch((Xbatch, ybatch), size=batchsize)
            updateparams!(storage, mod, Xmini, ymini)
        end
        mod.optparams.t += 1
        if verbose
            pred = At_mul_B(Xbatch, mod.modparams.weights) .+ mod.modparams.bias
            obj = value(mod.obj.loss, ybatch, pred, AvgMode.Mean()) + value(mod.obj.penalty, mod.modparams.weights)
            println("epoch $(mod.optparams.t - 1): objective = $obj")
        end
    end
end
