#pwd() = "qua5/experiemnts"
include("example/common/plotting.jl")
include("example/common/util.jl")



#plotting params
x = -7:.1:7
y = -7:.1:7
f = (x, y) -> exp(lpdf([x, y]))
#seting path
fig_path = "example/gaussian_mix/figure/"


#Contour for the target distribution
lpdf([5.,5.])
x = -7:.01:7
y = -7:.01:7
contourf(x, y, (x, y) -> exp(lpdf([x, y])))


#plot the annealing sched
x =  1:1000
pc =plot(x, i ->  cyclical_sched(i, 500, 5.), lw = 2, xtickfontsize=16,ytickfontsize=16,
            label = "cycle = 2", legend = :topleft)
plot!(x, i ->  cyclical_sched(i, 200, 5.), alpha = 0.7,lw = 2, label = "cycle = 4")
savefig(pc, joinpath(fig_path, "Asched_c.png"))



pp = plot(x, i ->  cyclical_sched(i, 500, .5), lw = 2, xtickfontsize=16,ytickfontsize=16,
        label = "p  = 0.5", legend = :topleft)
plot!(x, i ->  cyclical_sched(i, 500, 1.),lw = 2, label = "p = 1")
plot!(x, i ->  cyclical_sched(i, 500, 5.), lw = 2, label = "p = 5")
savefig(pp, joinpath(fig_path, "Asched_p.png"))