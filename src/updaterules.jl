@inline function updategrad!{T<:AbstractFloat}(
                                                grad::Vector{T},
                                                gradbias::Vector{T},
                                                storage::Vector{T}, #same size as ybatch
                                                obj::RegularizedLoss,
                                                Xbatch::AbstractMatrix{T},
                                                ybatch::AbstractVector{T},
                                                model::Model
                                                )
    batchsize = size(Xbatch, 2)

    # ∇{w}(L + P) = X'*((L)'(X'w + b, y))/N + (P)'(w)
    At_mul_B!(storage, Xbatch, model.weights)
    storage .+= model.bias
    deriv!(storage, obj.loss, ybatch, storage)
    A_mul_B!(grad, Xbatch, storage)
    grad ./= batchsize
    addgrad!(grad, obj.penalty, model.weights)
    gradbias .= mean(storage)
    grad, gradbias
end

#################### SGD UPDATE ###############################################

function updateparams!{T <: AbstractFloat}(storage::SGDStorage{T},
                                            mod::OnlineModel{T, <:Number, SGDOptimizer},
                                            Xmini::AbstractMatrix{T},
                                            ymini::Vector{T})

    grad, gradbias = updategrad!(storage.grad, storage.gradbias, storage.derv,
                                mod.obj, Xmini, ymini, mod.mod)
    η = T(mod.opt.newstepsize(mod.opt.t))
    mod.mod.weights .-= η .* grad
    mod.mod.bias .-= η .* gradbias
end
