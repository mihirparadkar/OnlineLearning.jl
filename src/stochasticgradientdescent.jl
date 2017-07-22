fit!{T <: DerivedOLM}(dolm::T) = fit!(dolm.mod)



function fit!{T}(mod, Xsamp::AbstractMatrix{T}, ysamp::Vector{T};
                epochs::Int=1,
                batchsize::Int=32,
                shuffle::Bool=true,
                verbose::Bool=true)

    niter = cld(size(Xsamp, 2), batchsize) * epochs
    if shuffle
        Xs, ys = shuffleobs((Xsamp, ysamp))
    else
        Xs, ys = Xsamp, ysamp
    end

    grad = zeros(T, size(Xs, 1))
    pred = zeros(T, batchsize)
    derv = zeros(T, batchsize)

    batchit = RandomBatches((Xs, ys), count=niter, size=batchsize)
    #Xtmp = zeros(size(Xsamp,1), batchsize)
    #ytmp = zeros(batchsize)
    for (Xbatch, ybatch) in BufferGetObs(batchit)

        grad, gradbias = updategrad!(grad, derv,
                                    mod.loss, mod.penalty,
                                    Xbatch, ybatch,
                                    mod.weights, mod.bias)

        η = T(mod.newstepsize(mod.t))
        mod.weights .-= η .* grad
        mod.bias -= η * gradbias

        pred .= At_mul_B!(pred, Xbatch, mod.weights) .+ mod.bias

        obj = value(mod.loss, ybatch, pred, AvgMode.Mean()) + value(mod.penalty, mod.weights)
        if verbose && (mod.t % (cld(niter, epochs)) == 0)
            println("epoch $(div(mod.t, cld(niter, epochs))): objective = $obj")
        end

        mod.t += 1
    end
end
