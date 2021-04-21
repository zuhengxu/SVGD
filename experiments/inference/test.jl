using SteinDiscrepancy

# import the kernel stein discrepancy function and kernel to use
using SteinDiscrepancy: SteinInverseMultiquadricKernel, ksd
# define the grad log density of standard normal target
function gradlogp(x::Array{Float64,1})
    -x
end

# grab sample
X = randn(500, 2)
# create the kernel instance
kernel = SteinGaussianKernel()
# compute the KSD2
result = ksd(points=X, gradlogdensity=x-> ForwardDiff.gradient(lpdf, x), kernel=kernel)
# get the final ksd
kernel_stein_discrepancy = sqrt(result.discrepancy2)



