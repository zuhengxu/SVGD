#pwd() = "qua5/experiments"


using Random, Distributions
##########
# 4*4 grid Gaussian mixture N(, 0.5^2)
##########

function logπ(Z::Vector, Mu::Array{Float64, 2})
    return logsumexp(-2. *sum((Z .- Mu).^2, dims = 1))
end


# the means for 16 Guassian components
Mu = ones(2, 16)
for j in 1:4
    @views for i in 1:4
        Mu[:, 4*(j-1) + i] = [-5. + 2*(i-1)*5. /3. , -5. + 2*(j-1)*5. /3. ]
    end
end

# lpdf for the Guassian mixture
lpdf = Z-> logπ(Z, Mu)
# grad for lpdf
∇lpdf = Z-> ∇logπ(Z, lpdf)
hessian_grid = Z -> ForwardDiff.hessian(lpdf, Z)


##########
# 2 component Gaussian mixture with different variance N(-5, 0.2^2) + N(5, 5^2)
##########
function log_mix(Z::Vector)
    return  logsumexp([logpdf(MvNormal([5.,5.], 0.5), Z), logpdf(MvNormal([-5., -5.], 3), Z) ])
end

∇log_mix = Z-> ∇logπ(Z, log_mix)
hessian_mix = Z -> ForwardDiff.hessian(log_mix, Z)