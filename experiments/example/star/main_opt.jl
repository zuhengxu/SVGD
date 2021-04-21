#pwd() = "qua5/experiemnts"
include("inference/SVGD.jl")
include("example/common/plotting.jl")
include("example/star/model.jl")
include("example/common/util.jl")


##########
# set up SVGD
##########
Random.seed!(2021);
# init ptcs
x0  = rand(Normal(0,1), (2,500))
multi_gaussian_SV = SVGD(x0, lpdf, ∇lpdf, RBFkernel!, i-> 1.)
#plotting params
x = -3:.1:3
y = -3:.1:3
f = (s, t)-> exp(0.5*logπ_multi_gaussian([s t]))
#seting path
fig_path = "example/star/figure/"


##########
##########
# 1. get trace with various update rules
# 2. generate gif to see the moving traces
##########
##########

#GD update
gd = GD(lrt = i-> .1, niters = 200)
T_gd = svgd_trace(multi_gaussian_SV, gd, 100, 0.5);
gif2d(x, y, f, T_gd; folder = fig_path, name = "gd.gif");

#Momentum
mom= Momentum(lrt = i-> .1, niters = 200)
T_mom = svgd_trace(multi_gaussian_SV, mom, 100, 0.5);
gif2d(x, y, f, T_mom; folder = fig_path, name = "mom.gif");

# AdaGrad
ada = AdaGrad(lrt = i-> .15, niters = 200)
T_ada = svgd_trace(multi_gaussian_SV, ada, 100, 0.5);
gif2d(x, y, f, T_ada; folder = fig_path, name = "ada.gif");

#RMSprop
rms = RMSprop(lrt = i-> .1, niters = 200)
T_rms =  svgd_trace(multi_gaussian_SV, rms, 100, 0.5);
gif2d(x, y, f, T_rms; folder = fig_path,name = "rms.gif");

#Riemanian accelerate method
wag = WAG(lrt = i-> 0.22/(1.0 + i^0.12), niters = 200, α= 3.6) 
T_wag = svgd_trace(multi_gaussian_SV, wag, 100, 0.5);
gif2d(x, y, f, T_wag; folder = fig_path, name = "wag.gif");


#Wasserstein Nes (bit hard to tune)
wnes  = WNes(lrt = i-> .15, niters =  200, u = 3, b = 0.5)
T_nes = svgd_trace(multi_gaussian_SV, wnes, 100, 0.5);
gif2d(x, y, f, T_nes; folder = fig_path, name = "wnes.gif");


##########
# saving all the traces
##########
save("example/star/result/all_traces.jld", 
    "gd", T_gd, "mom", T_mom, "rms", T_rms, "wag", T_wag, "nes", T_nes)



