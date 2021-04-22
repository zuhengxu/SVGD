using JLD
# computing KSD when they cover the modes
using SteinDiscrepancy
using SteinDiscrepancy: ksd

# compute the ksd using gaussian kernel
# only works for posterior on full support
function ksd_gaussian(T, grd::Function)
    result = ksd(points = permutedims(T, (2,1)), gradlogdensity= grd, kernel=SteinGaussianKernel())
    return sqrt(result.discrepancy2)
end

# ksd along the iterations
function ksd_trace(T::Array{Float64, 3}, grd::Function)
    f  = M ->  ksd_gaussian(M, grd)
    ksdseq = map(f, eachslice(T, dims = 3))
    return ksdseq
end

# elbo estimation to show the change of kl
function elbo(T, lpdf)
    el = mean(map(lpdf, eachslice(T, dims = 2)))
    return el
end

# elbo along the iterations
function elbo_trace(T::Array{Float64, 3}, lpdf)
    f = M-> elbo(M, lpdf)
    es = map(f, eachslice(T, dims =3))
    return es
end


# project 2d scatters to the line y=x
function proj_xy(T::Array{Float64, 2})
    a = mapslices(mean, T, dims = 1)
    return dropdims(a;dims = 1) 
end
