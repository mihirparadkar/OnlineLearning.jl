import LearnBase: value, deriv
using LossFunctions

# log1pexp: log(1+exp(x))
#
log1pexp(x::Real) = x < 18.0 ? log1p(exp(x)) : x < 33.3 ? x + exp(-x) : oftype(exp(-x), x)
log1pexp(x::Float32) = x < 9.0f0 ? log1p(exp(x)) : x < 16.0f0 ? x + exp(-x) : oftype(exp(-x), x)
const softplus = log1pexp

## logsumexp

function logsumexp{T<:Real}(x::T, y::T)
    x == y && abs(x) == Inf && return x
    x > y ? x + log1p(exp(y - x)) : y + log1p(exp(x - y))
end

logsumexp(x::Real, y::Real) = logsumexp(promote(x, y)...)

function logsumexp{T<:Real}(x::AbstractArray{T})
    S = typeof(exp(zero(T)))    # because of 0.4.0
    isempty(x) && return -S(Inf)
    u = maximum(x)
    abs(u) == Inf && return any(isnan, x) ? S(NaN) : u
    s = zero(S)
    for i = 1:length(x)
        @inbounds s += exp(x[i] - u)
    end
    log(s) + u
end

## softmax

function softmax!{R<:AbstractFloat,T<:Real}(r::AbstractArray{R}, x::AbstractArray{T})
    n = length(x)
    length(r) == n || throw(DimensionMismatch("Inconsistent array lengths."))
    u = maximum(x)
    s = 0.
    @inbounds for i = 1:n
        s += (r[i] = exp(x[i] - u))
    end
    invs = convert(R, inv(s))
    @inbounds for i = 1:n
        r[i] *= invs
    end
    r
end

softmax!{T<:AbstractFloat}(x::AbstractArray{T}) = softmax!(x, x)
softmax{T<:Real}(x::AbstractArray{T}) = softmax!(Array{Float64}(size(x)), x)

# Abstract Types ##############################################################
abstract type CategoricalLoss <: SupervisedLoss end

#multinomiallogit+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
struct MultinomialLogitLoss <: CategoricalLoss
  N::Int
end

function LearnBase.value(loss::MultinomialLogitLoss, target::Unsigned, output::Vector{O} where O <: Number)
  probtarg = output[target]
  logsumexp(output) - probtarg
end

function LearnBase.deriv!{O <: AbstractFloat}(dest::AbstractVector{O}, loss::MultinomialLogitLoss, target::Unsigned, output::AbstractVector{O})
  softmax!(dest, output)
  ret[target] .-= 1
  ret
end


#multiclassl1hinge+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
struct MulticlassL1HingeLoss <: CategoricalLoss
  N::Int
end

function LearnBase.value{O <: AbstractFloat}(loss::MulticlassL1HingeLoss, target::Unsigned, output::AbstractVector{O})
  y = target
  maxnoty = O(-Inf)
  @inbounds for i in eachindex(output)
    if i != y
      @inbounds maxnoty = max(output[i], maxnoty)
    end
  end
  max(0, 1 + maxnoty - output[y])
end

function LearnBase.deriv!{O <: AbstractFloat}(dest::AbstractVector{O}, loss::MulticlassL1HingeLoss, target::Unsigned, output::AbstractVector{O})
  y = target
  maxnoty = O(-Inf)
  indmaxnoty = 0
  @inbounds for i in eachindex(output)
    if i != y
      @inbounds output[i] > maxnoty && (indmaxnoty = i)
      @inbounds maxnoty = max(output[i], maxnoty)
    end
  end
  scale!(dest, 0)
  if output[y] <= maxnoty + 1
    ret[y] = -1
    ret[indmaxnoty] = 1
  end
  ret
end


#ovrloss++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
struct OVRLoss{L <: MarginLoss} <: CategoricalLoss
  loss::L
  N::Int
end

function LearnBase.value{O <: AbstractFloat}(loss::OVRLoss, target::Unsigned, output::AbstractVector{O})
  ret = zero(O)
  for i in eachindex(output)
    if i != target
      @inbounds ret += value(loss.loss, -1, output[i])
    else
      @inbounds ret += value(loss.loss, 1, output[i])
    end
  end
  ret
end

function LearnBase.deriv!{O <: AbstractFloat}(dest::AbstractVector{O}, loss::OVRLoss, target::Unsigned, output::AbstractVector{O})
  scale!(dest, 0)
  for i in eachindex(output)
    if i != target
      @inbounds ret[i] = deriv(loss.loss, -1, output[i])
    else
      @inbounds ret[i] = deriv(loss.loss, 1, output[i])
    end
  end
  ret
end
