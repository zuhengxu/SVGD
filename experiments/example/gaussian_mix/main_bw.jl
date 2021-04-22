#pwd() = "qua5/experiemnts"
include("inference/SVGD.jl")
include("example/gaussian_mix/model.jl")
include("example/common/result.jl")
include("example/common/util.jl")
include("example/common/plotting.jl")





##########
##########
# 1. ASVGD with different bw 
##########
##########

##########
#set up the A-SVGD
##########
Random.seed!(2021);
# init prtc
x0  = rand(Normal(0,0.5), (2,500));
ASV = SVGD(x0, lpdf, ∇lpdf, RBFkernel!, i->  cyclical_sched(i, 500, 1.));
#RMSprop update rule
niters = 1000
rms = RMSprop(lrt = i-> .05, niters = niters)

#plotting params
x = -7:.1:7
y = -7:.1:7
f = (x, y) -> exp(lpdf([x, y]))
#figure path
fig_path = "example/gaussian_mix/figure/"


##########
# run ASVGD and obtain plots
##########
# get the trace
Tmed = svgd_trace(ASV, rms, 100, -1.);
T1 = svgd_trace(ASV, rms, 100, 1.);
T5 = svgd_trace(ASV, rms, 100, 5.);
T10 = svgd_trace(ASV, rms, 100, 10.);

# compute ksds 
# ksd_med =  ksd_gaussian(permutedims(Tmed[:,:,end], (2,1)), x-> ForwardDiff.gradient(lpdf, x))
# ksd_1 = ksd_gaussian(permutedims(T1[:,:,end], (2,1)), x-> ForwardDiff.gradient(lpdf, x))
# ksd_5 = ksd_gaussian(permutedims(T5[:,:,end], (2,1)), x-> ForwardDiff.gradient(lpdf, x))
# ksd_10 = ksd_gaussian(permutedims(T10[:,:,end], (2,1)), x-> ForwardDiff.gradient(lpdf, x))

# save traces and contour plots
save("example/gaussian_mix/result/bw_result.jld", "bwmed", Tmed, "bw1", T1, "bw5", T5, "bw10", T10)
contour_save(x, y, f, Tmed[:,:,end]; folder = fig_path, name = "bwmed.png", title = "bw = median , KSD = "*string(round(ksd_10, digits = 3)))
contour_save(x, y, f, T1[:,:,end]; folder = fig_path, name = "bw1.png", title = "bw = 1, KSD = "*string(round(ksd_1, digits = 3)))
contour_save(x, y, f, T5[:,:,end]; folder = fig_path, name = "bw5.png", title = "bw = 5, KSD = "*string(round(ksd_5, digits = 3)))
contour_save(x, y, f, T10[:,:,end]; folder = fig_path, name = "bw10.png", title = "bw = 10, KSD = "*string(round(ksd_10, digits = 3)))

# result = ksd(points=permutedims(T_rms_anneal, (2,1)), gradlogdensity=x-> ForwardDiff.gradient(lpdf, x), kernel=SteinGaussianKernel())
# ksd_med = sqrt(result.discrepancy2)





##########
##########
# 2. Plots to show that better use different bw for different points 
##########
##########
Random.seed!(2021);
# init prtc
x0  = rand(Normal(0,5), (2,300));
ASV = SVGD(x0, log_mix, ∇log_mix, RBFkernel!, i->  cyclical_sched(i, 500, 1.));
#RMSprop update rule
niters = 1000
rms = RMSprop(lrt = i-> .1, niters = niters)

#plotting params
x = -15:.1:8
y = -15:.1:8
f = (x, y) -> exp(0.35*log_mix([x, y]))

##########
# run ASVGD and obtain plots
##########
# get the trace
T_small = svgd_trace(ASV, rms, 100, .1);
T_large = svgd_trace(ASV, rms, 100, 10.);

#compute ksd
ksd_small = ksd_gaussian(permutedims(T_small[:,:,end], (2,1)), x-> ForwardDiff.gradient(log_mix, x)) # 0.073
ksd_large = ksd_gaussian(permutedims(T_large[:,:,end], (2,1)), x-> ForwardDiff.gradient(log_mix, x)) #0.071

# save traces and contour plots
save("example/gaussian_mix/result/bw_result_mix.jld", "bw_large", T_large, "bw_small", T_small)
contour_save(x, y, f, T_small[:,:,end]; folder = fig_path, name = "bwsmall.png", title = "bw = 0.1")
contour_save(x, y, f, T_large[:,:,end]; folder = fig_path, name = "bwlarge.png", title = "bw = 10")
gif2d(x, y, f, T_small; folder = fig_path, name = "bwsmall.gif")
gif2d(x, y, f, T_large; folder = fig_path, name = "bwlarge.gif")

# look at the slice y = x
π = Z -> 0.5*pdf(MvNormal([5.,5.], 0.5), Z)+0.5*pdf(MvNormal([-5., -5.], 3), Z) 
hist_save(x, z-> π([z, z]), proj_xy(T_small[:,:,end]); folder = fig_path, name = "slice_small.png", title = "bw = 0.1")
hist_save(x, z-> π([z, z]), proj_xy(T_large[:,:,end]); folder = fig_path, name = "slice_large.png", title = "bw = 10")
