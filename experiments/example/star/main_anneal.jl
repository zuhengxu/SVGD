#pwd() = "qua5/experiemnts"
include("inference/SVGD.jl")
include("example/common/plotting.jl")
include("example/common/util.jl")
include("example/star/model.jl")

##########
##########
# See the entropy regularizer
##########
##########


##########
# set up SVGD
##########
Random.seed!(2021);
# init ptcs
x0  = rand(Normal(0,1), (2,500))

#plotting params
x = -3:.1:3
y = -3:.1:3
f = (s, t)-> exp(0.5*logπ_multi_gaussian([s t]))
#seting path
fig_path = "example/star/figure/"


SV = SVGD(x0, lpdf, ∇lpdf, RBFkernel!, i-> .5)
rms = RMSprop(lrt = i-> 2. , niters = 500)
T =  svgd_trace(SV, rms, 100, 0.5);
contour_save(x, y, f, T[:,:,end]; folder = fig_path,name = "rms_boost.png")

