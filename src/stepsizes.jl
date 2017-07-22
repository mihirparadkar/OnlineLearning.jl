"""
constant step size
"""
function conststepsize(eta0::Number)
    return (t::Int64 -> eta0)
end

"""
step size inversely proportional to t^power_t
"""
function invstepsize(eta0::Number, power_t::Number)
    return (t::Int64 -> eta0 / (t ^ power_t))
end
