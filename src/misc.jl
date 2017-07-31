#If y is a Vector, it turns it into a RowVector
function correctdims(y::DenseVector)
    y.'
end

#If y is a RowVector or a matrix, preserves the dimensions
function correctdims(y::AbstractMatrix)
    y
end

function deriv!(dest::AbstractMatrix, loss::Loss, target::AbstractMatrix, output::AbstractMatrix)
    deriv!(reshape(dest, :), loss, reshape(target, :), reshape(output, :))
end
