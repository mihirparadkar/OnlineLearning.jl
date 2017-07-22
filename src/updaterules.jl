@inline function updategrad!{T<:AbstractFloat}(
                                                grad::Vector{T},
                                                storage::Vector{T}, #same size as ybatch
                                                obj::RegularizedLoss,
                                                Xbatch::AbstractMatrix{T},
                                                ybatch::AbstractVector{T},
                                                model::ModelParams
                                                )
    batchsize = size(Xbatch, 2)

    # ∇{w}(L + P) = X'*((L)'(X'w + b, y))/N + (P)'(w)
    At_mul_B!(storage, Xbatch, model.weights)
    storage .+= model.bias
    deriv!(storage, obj.loss, ybatch, storage)
    A_mul_B!(grad, Xbatch, storage)
    grad ./= batchsize
    addgrad!(grad, obj.penalty, model.weights)
    gradbias = mean(storage)
    grad, gradbias
end

#################### SGD UPDATE ###############################################

function updateparams!{T <: AbstractFloat}(storage::SGDStorage{T},
                                            mod::OnlineLinearModel{T, SGDOptimizer},
                                            Xmini::AbstractMatrix{T},
                                            ymini::Vector{T})

    grad, gradbias = updategrad!(storage.grad, storage.derv,
                                mod.obj, Xmini, ymini, mod.modparams)
    η = T(mod.optparams.newstepsize(mod.optparams.t))
    mod.modparams.weights .-= η .* grad
    mod.modparams.bias -= η * gradbias
end

####################### ADAGRAD UPDATE #########################################

function updateparams!{T <: AbstractFloat}(storage::AdaGradStorage{T},
                                            mod::OnlineLinearModel{T, AdaGradOptimizer{T}},
                                            Xmini::AbstractMatrix{T},
                                            ymini::Vector{T})

    grad, gradbias = updategrad!(storage.grad, storage.derv,
                                mod.obj, Xmini, ymini, mod.modparams)
    η = T(mod.optparams.newstepsize(mod.optparams.t))
    Binvsq = mod.optparams.Binvsq
    Binvsq .+= grad .^ 2
    mod.optparams.avgsqbias += gradbias ^ 2
    mod.modparams.weights .-= η .* grad ./ sqrt.(Binvsq)
    mod.modparams.bias -= η * gradbias / sqrt(mod.optparams.avgsqbias)
end
