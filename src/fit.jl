function defaultbatchsize(numobs::Int)
    #The nearest power of 2 above the square root of the number of observations
    numobs |> sqrt |> log2 |> ceil |> exp2 |> Int
end

function decodelabels{D<:AbstractFloat,L<:Number}(mod::OnlineModel{D,L},
                                                        ysamp::DenseArray{L})
    convert(Array{D}, ysamp)
end

function decodelabels{D<:AbstractFloat,L<:Bool}(mod::OnlineModel{D,L},
                                                ysamp::DenseArray{L})
    D[2el - 1 for el in ysamp]
end

function decodelabels{D<:AbstractFloat,L<:Unsigned}(mod::OnlineModel{D,L},
                                                    ysamp::DenseArray{L})
    ysamp
end

function partialfit!{T <: AbstractFloat}(mod::OnlineModel{T,<:Number,<:Optimizer},
                                X::AbstractMatrix{T}, y::DenseArray;
                                shuffle::Bool=true,
                                batchsize::Int=defaultbatchsize(size(X,2)),
                                epochs::Int=1,
                                verbose::Bool=false)

    yencoded = decodelabels(mod, y)
    nfeats = size(X, 1)
    if shuffle
        Xbatch, ybatch = map(Array, shuffleobs((X, yencoded)))
    else
        Xbatch, ybatch = X, yencoded
    end

    storage = allocate_storage(mod.mod.weights, batchsize, mod.opt)

    for iter in 1:epochs
        for (Xmini, ymini) in eachbatch((Xbatch, ybatch), size=batchsize)
            updateparams!(storage, mod, Xmini, ymini)
        end
        mod.opt.t += 1
        if verbose
            pred = decision_func(mod, Xbatch)
            obj = value(mod.obj.loss, ybatch, pred, AvgMode.Mean()) + value(mod.obj.penalty, mod.mod.weights)
            println("epoch $(mod.opt.t - 1): objective = $obj")
        end
    end
end
