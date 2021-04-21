using JLD
# computing KSD when they cover the modes
using SteinDiscrepancy
using SteinDiscrepancy: ksd


# t = 15
# z = [1,3]
# # creats a file saves all traces--named in a dictionary
# save("example/gaussian_mix/test/myfile.jld", "t", t, "arr", z)
# # load that file
# d = load("example/gaussian_mix/test/myfile.jld")


# compute the ksd using gaussian kernel
function ksd_gaussian(T::Array{Float64, 2}, grd::Function)
    result = ksd(points = T, gradlogdensity= grd, kernel=SteinGaussianKernel())
    return sqrt(result.discrepancy2)
end

# # moment estimates given traces
# function mean_est(T::Array{Float64, 3})
#     return dropdims(mean(T, dims =2);dims = 2)
# end

# function var_est(T::Array{Float64, 3})
#     return 
# end

# project 2d scatters to the line y=x
function proj_xy(T::Array{Float64, 2})
    a = mapslices(mean, T, dims = 1)
    return dropdims(a;dims = 1) 
end
