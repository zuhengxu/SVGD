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
# 3. using local bw with curvature_scaling for gaussian mix
##########
##########
Random.seed!(2021);
x0  = rand(Normal(0,5), (2,300));
ASV_local = SVGD_bw(init_ptc = x0, lpdf = log_mix, ∇lpdf =  ∇log_mix, 
            Hessian = hessian_mix, anneal = i->  cyclical_sched(i, 500, 1.));
niters = 1000
rms = RMSprop(lrt = i-> .1, niters = niters)
#plotting params
x = -15:.1:8
y = -15:.1:8
f = (x, y) -> exp(0.35*log_mix([x, y]))
T_local = svgd_trace(ASV_local, rms, 100)

ksd_local = ksd_gaussian(permutedims(T_local[:,:,end], (2,1)), x-> ForwardDiff.gradient(log_mix, x))# 0.092
gif2d(x, y, f, T_local; folder = fig_path, name = "bwlocal.gif")
contour_save(x, y, f, T_local[:,:,end]; folder = fig_path, name = "bwlocal.png", title = "")

# plot the bw value surface
p1 = plot(-3:.1:8, -3:.1:8, (x, y)-> log10(curvature_scaling([x, y], hessian_mix)), st=:surface, 
    colorbar = :none,   title = "Local Bandwidth", xtickfontsize=18,ytickfontsize=18, ztickfontsize = 18)
savefig(p1, joinpath(fig_path, "bwsurf_mix.png"))

# plot the projected points along y = x 
π = Z -> (0.5*pdf(MvNormal([5.,5.], 0.5), Z)+0.5*pdf(MvNormal([-5., -5.], 3), Z))
hist_save(x, z-> π([z, z]), proj_xy(T_local[:,:,end]); folder = fig_path, name = "slice_local.png", title = "Local bw")

##########
# 4. using local bw with curvature_scaling for gaussian grid
##########

Random.seed!(2021);
x0  = rand(Normal(0,0.5), (2,500));
x10 = rand(Normal(10, 0.5), (2, 500));
ASV_local10 = SVGD_bw(init_ptc = x10, lpdf = lpdf, ∇lpdf =  ∇lpdf, 
            Hessian = hessian_grid, anneal = i->  cyclical_sched(i, 500, 5.));
ASV_local = SVGD_bw(init_ptc = x0, lpdf = lpdf, ∇lpdf =  ∇lpdf, 
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
T_grid_local = svgd_trace(ASV_local, rms, 100)
T_grid_local10 = svgd_trace(ASV_local10, rms, 100)
# compute ksds 
ksd_local = ksd_gaussian(permutedims(T_grid_local[:,:,end], (2,1)), x-> ForwardDiff.gradient(lpdf, x))

gif2d(x, y, f, T_grid_local; folder = fig_path, name = "grid_local.gif")
contour_save(x, y, f, T_grid_local[:,:,end]; folder = fig_path, name = "grid_local.png")
contour_save(x, y, f, T_grid_local10[:,:,end]; folder = fig_path, name = "grid_local10.png")    
##plot the bw value on this fig
p2 = plot(-7:.1:7, -7:.1:7, (x,y )-> log10(curvature_scaling([x, y], hessian_grid)), 
        colorbar = :none, st = :surface, title = "Local Bandwidth", 
        xtickfontsize=18,ytickfontsize=18, ztickfontsize  =18)
savefig(p2, joinpath(fig_path, "bwsurf_grid.png"))