########################### GRADIENT UPDATE #########################################

@inline function updategrad!{T<:AbstractFloat}(
                                                grad::Vector{T},
                                                gradbias::Vector{T},
                                                storage::Vector{T}, #same size as ybatch
                                                obj::RegularizedLoss,
                                                Xbatch::AbstractMatrix{T},
                                                ybatch::DenseVector{T},
                                                weights::Vector{T},
                                                bias::Vector{T}
                                                )
    batchsize = size(Xbatch, 2)

    # ∇{w}(L + P) = X'*((L)'(X'w + b, y))/N + (P)'(w)
    At_mul_B!(storage, Xbatch, weights)
    storage .+= bias
    deriv!(storage, obj.loss, ybatch, storage)
    A_mul_B!(grad, Xbatch, storage)
    grad ./= batchsize
    addgrad!(grad, obj.penalty, weights)
    gradbias .= mean(storage)
    grad, gradbias
end

@inline function updategrad!{T<:AbstractFloat}(
                                                grad::Matrix{T},
                                                gradbias::Vector{T},
                                                storage::Matrix{T},
                                                obj::RegularizedLoss,
                                                Xbatch::AbstractMatrix{T},
                                                ybatch::AbstractArray,
                                                weights::Matrix{T},
                                                bias::Vector{T}
                                                )
    batchsize = size(Xbatch, 2)

    # ∇{w}(L + P) = X'*((L)'(X'w + b, y))/N + (P)'(w)
    A_mul_B!(storage, weights, Xbatch)
    storage .+= bias
    deriv!(storage, obj.loss, ybatch, storage)
    A_mul_Bt!(grad, storage, Xbatch)
    grad ./= batchsize
    addgrad!(grad, obj.penalty, weights)
    gradbias .= vec(mean(storage, 2))
    grad, gradbias
end

#################### SGD UPDATE ###############################################

function updateparams!{T <: AbstractFloat}(storage::SGDStorage{T},
                                            mod::OnlineModel{T, <:Number, <:SGDOptimizer},
                                            Xmini::AbstractMatrix{T},
                                            ymini::Array)

    grad, gradbias = updategrad!(storage.grad, storage.gradbias, storage.derv,
                                mod.obj, Xmini, ymini, mod.mod.weights, mod.mod.bias)
    opt = mod.opt
    η = opt.η0 / (1 + opt.decay * opt.t^opt.power_t)
    opt.prevgrad .= η.*grad .+ opt.momentum .* opt.prevgrad
    opt.prevgradbias .= η.*gradbias .+ opt.momentum .* opt.prevgradbias
    mod.mod.weights .-= opt.prevgrad
    mod.mod.bias .-= opt.prevgradbias
end

################## NESTEROV UPDATE #################################################

function updateparams!{T <: AbstractFloat}(storage::NesterovStorage{T},
                                            mod::OnlineModel{T, <:Number, <:NesterovOptimizer},
                                            Xmini::AbstractMatrix{T},
                                            ymini::Array)

    opt = mod.opt
    η = opt.η0 / (1 + opt.decay * opt.t^opt.power_t)
    storage.nextweights .= mod.mod.weights .- opt.momentum .* opt.prevgrad
    storage.nextbias .= mod.mod.bias .- opt.momentum .* opt.prevgradbias
    grad, gradbias = updategrad!(storage.grad, storage.gradbias, storage.derv,
                                mod.obj, Xmini, ymini, storage.nextweights, storage.nextbias)
    opt.prevgrad .= η .* grad .+ opt.momentum .* opt.prevgrad
    opt.prevgradbias .= η .* gradbias .+ opt.momentum .* opt.prevgradbias
    mod.mod.weights .-= opt.prevgrad
    mod.mod.bias .-= opt.prevgradbias
end

########################### ADAGRAD UPDATE ##########################################

function updateparams!{T <: AbstractFloat}(storage::AdagradStorage{T},
                                            mod::OnlineModel{T, <:Number, <:AdagradOptimizer},
                                            Xmini::AbstractMatrix{T},
                                            ymini::Array)
    grad, gradbias = updategrad!(storage.grad, storage.gradbias, storage.derv,
                                mod.obj, Xmini, ymini, mod.mod.weights, mod.mod.bias)
    opt = mod.opt
    η = opt.η0 / (1 + opt.decay * opt.t^opt.power_t)
    opt.sqgrads .+= grad .^ 2
    opt.sqbiasgrads .+= gradbias .^ 2
    mod.mod.weights .-= η .* grad ./ sqrt.(opt.sqgrads .+ opt.ϵ )
    mod.mod.bias .-= η .* gradbias ./ sqrt.(opt.sqbiasgrads .+ opt.ϵ)
end
