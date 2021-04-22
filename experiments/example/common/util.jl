using ForwardDiff
using Base.Threads: @threads



# cannot specify the type of w as array
# otherwise cannot ForwardDiff
function logsumexp(w) 
    max = maximum(w)
    we = expm1.(w .- max) .+ 1.
    return max  + log1p(sum(we) - 1.)
end


# stable softmax function
function softmax(W)
    Ws = W .- maximum(W)
    return (expm1.(Ws) .+1.) / (expm1(logsumexp(Ws)) + 1.)
end



# gradient of logπ applied to all datapoints
function ∇logπ(Z::Array{Float64, 2}, lpdf::Function)
    ∇lpdf = x-> ForwardDiff.gradient(lpdf, x)
    # threads: macros to boost performance of for loop
    # inbounds: close the bound check for array
    G = similar(Z)
    @threads for i in 1:size(Z,2)
    @inbounds G[:,i] .= ∇lpdf(Z[:,i])
    end
    return G
end

