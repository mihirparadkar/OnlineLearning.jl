function decision_function!{T}(ypred::Vector{T}, om::OnlineModel, Xpred::AbstractMatrix{T})
    At_mul_B!(ypred, Xpred, om.mod.weights)
    ypred .+= om.mod.bias
    ypred
end

function decision_function{T}(om::OnlineModel, Xpred::AbstractMatrix{T})
    ypred = zeros(T, size(Xpred, 2))
    decision_function!(ypred, om, Xpred)
end

function predict{T<:Number, L<:Bool}(om::OnlineModel{T,L,<:Optimizer}, Xpred::AbstractMatrix{T}; threshold=0.0)
    ypred = decision_function(om, Xpred)
    out = BitArray{1}(size(Xpred, 2))
    @inbounds for i in eachindex(out)
        out[i] = ypred[i] > threshold
    end
    out
end

function predict{T<:Number, L<:AbstractFloat}(om::OnlineModel{T,L,<:Optimizer}, Xpred::AbstractMatrix{T})
    decision_function(om, Xpred)
end

function predict{T<:Number, L<:Integer}(om::OnlineModel{T,L,<:Optimizer}, Xpred::AbstractMatrix{T}, nlevels)
    ypred = decision_function(om, Xpred)
    ypred .= round.(clamp!(ypred, 1, nlevels))
    Array{L}(ypred)
end
