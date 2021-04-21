#pwd() = "qua5/experiemnts"
include("inference/SVGD.jl")
include("example/gaussian_mix/model.jl")
include("example/common/result.jl")
include("example/common/util.jl")
include("example/common/plotting.jl")



#figure path
fig_path = "example/gaussian_mix/figure/"
##########
##########
# 5. using localized Gauss kernel for gaussian mix
##########
##########
Random.seed!(2021);
x0  = rand(Normal(0,5), (2,300));
ASV_Gauss = SVGD_Gauss(init_ptc = x0, lpdf = log_mix, ∇lpdf =  ∇log_mix, 
            Hessian = hessian_mix, anneal = i->  cyclical_sched(i, 500, 1.));
niters = 1000
rms = RMSprop(lrt = i-> .1, niters = niters)
#plotting params
x = -15:.1:8
y = -15:.1:8
f = (x, y) -> exp(0.35*log_mix([x, y]))
T_Gauss = svgd_trace(ASV_Gauss, rms, 100)

ksd_gauss = ksd_gaussian(permutedims(T_Gauss[:,:,end], (2,1)), x-> ForwardDiff.gradient(log_mix, x))# 0.092
gif2d(x, y, f, T_Gauss; folder = fig_path, name = "mixGauss.gif")
contour_save(x, y, f, T_Gauss[:,:,end]; folder = fig_path, name = "mixGauss.png", title = "")


# plot the projected points along y = x 
π = Z -> (0.5*pdf(MvNormal([5.,5.], 0.5), Z)+0.5*pdf(MvNormal([-5., -5.], 3), Z))
hist_save(x, z-> π([z, z]), proj_xy(T_Gauss[:,:,end]); folder = fig_path, name = "slice_gauss.png", title = "Localized kernel")



##########
# 6. using localized Gauss kernel for gaussian grid
##########

Random.seed!(2021);
x0  = rand(Normal(0,0.5), (2,500));
x10 = rand(Normal(10, 0.5), (2, 500));
ASV_gs10 = SVGD_Gauss(init_ptc = x10, lpdf = lpdf, ∇lpdf =  ∇lpdf, 
            Hessian = hessian_grid, anneal = i->  cyclical_sched(i, 500, 5.));
ASV_gs = SVGD_Gauss(init_ptc = x0, lpdf = lpdf, ∇lpdf =  ∇lpdf, 
            Hessian = hessian_grid, anneal = i->  cyclical_sched(i, 500, 5.));
niters = 1000
rms = RMSprop(lrt = i-> .05, niters = niters)
#plotting params
x = -7:.1:7
y = -7:.1:7
f = (x, y) -> exp(lpdf([x, y]))

##########
# run SVGD_bw and obtain plots
##########
T_grid_gs = svgd_trace(ASV_gs, rms, 100)
T_grid_gs10 = svgd_trace(ASV_gs10, rms, 100)
# compute ksds 
ksd_gs = ksd_gaussian(permutedims(T_grid_gs[:,:,end], (2,1)), x-> ForwardDiff.gradient(lpdf, x))

gif2d(x, y, f, T_grid_gs; folder = fig_path, name = "grid_gs.gif")
contour_save(x, y, f, T_grid_gs[:,:,end]; folder = fig_path, name = "grid_gs.png")
contour_save(x, y, f, T_grid_gs10[:,:,end]; folder = fig_path, name = "grid_gs10.png")    
