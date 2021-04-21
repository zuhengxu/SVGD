##########
# star shape distribution: a 2d Gaussian mixture 
##########
function logπ_multi_gaussian(z)
    g1 = -0.5*z[1]^2/.15^2 - 0.5*z[2]^2/1^2
    g2 = -0.5*z[1]^2/1^2 - 0.5*z[2]^2/.15^2
    m = max(g1,g2)
    return m + log(exp(g1-m) + exp(g2-m))
end


lpdf = logπ_multi_gaussian
∇lpdf = Z-> ∇logπ(Z, lpdf)

