function decision_func!{T}(ypred::Vector{T}, om::OnlineModel, Xpred::AbstractMatrix{T})
    At_mul_B!(ypred, Xpred, om.mod.weights)
    ypred .+= om.mod.bias
    ypred
end

function decision_func!{T}(ypred::Matrix{T}, om::OnlineModel, Xpred::AbstractMatrix{T})
    A_mul_B!(ypred, om.mod.weights, Xpred)
    ypred .+= om.mod.bias
    ypred
end

function decision_func{T}(om::OnlineModel{T,<:Number,<:Optimizer,1}, Xpred::AbstractMatrix{T})
    ypred = zeros(T, size(Xpred, 2))
    decision_func!(ypred, om, Xpred)
end

function decision_func{T <: Unsigned}(om::OnlineModel{T,<:Number,<:Optimizer,2}, Xpred::AbstractMatrix{T})
    ypred = zeros(T, (om.enc.nlevels, size(Xpred, 2)))
    decision_func!(ypred, om, Xpred)
end

function decision_func{T}(om::OnlineModel{T,<:Number,<:Optimizer,2}, Xpred::AbstractMatrix{T})
    ypred = zeros(T, (size(om.mod.weights, 1), size(Xpred, 2)))
    decision_func!(ypred, om, Xpred)
end

function predict{T<:Number, L<:Bool}(om::OnlineModel{T,L,<:Optimizer}, Xpred::AbstractMatrix{T}; threshold=0.0)
    ypred = decision_func(om, Xpred)
    out = BitArray(size(ypred))
    @inbounds for i in eachindex(out)
        out[i] = ypred[i] > threshold
    end
    out
end

function predict{T<:Number, L<:AbstractFloat}(om::OnlineModel{T,L,<:Optimizer}, Xpred::AbstractMatrix{T})
    decision_func(om, Xpred)
end

function predict{T<:Number, L<:Integer}(om::OnlineModel{T,L,<:Optimizer}, Xpred::AbstractMatrix{T})
    ypred = decision_func(om, Xpred)
    ypred .= round.(clamp!(ypred, 1, om.enc.nlevels))
    Array{L}(ypred)
end

function predict{T<:Number, L<:Unsigned}(om::OnlineModel{T,L,<:Optimizer}, Xpred::AbstractMatrix{T})
    ypred = decision_func(om, Xpred)
    out = Array{L}(size(Xpred, 2))
    for i in eachindex(out)
        out[i] = indmax(view(ypred, :, i))
    end
    out
end

function loss{T<:Number}(om::OnlineModel{T}, target::AbstractArray, output::AbstractArray{T})
    value(om.opt.loss, target, output, AvgMode.Sum())
end
