#=
function LearnBase.deriv!(dest::AbstractMatrix, loss::Loss, target::AbstractMatrix, output::AbstractMatrix)
    deriv!(reshape(dest, :), loss, reshape(target, :), reshape(output, :))
end
=#
