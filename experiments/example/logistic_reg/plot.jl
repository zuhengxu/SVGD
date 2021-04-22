#pwd() = "qua5/experiemnts"
include("example/common/result.jl")
include("example/common/util.jl")
include("example/common/plotting.jl")



result_path = "example/logistic_reg/result/"
fig_path = "example/logistic_reg/figure/"

#load KSD and ELBO
K = load(joinpath(result_dir, "ksd.jld"))
E = load(joinpath(result_dir, "elbo.jld"))
ksd_seq = [ K["sv"] K["sv_local"] ]
elbo_seq = [ E["sv"] E["sv_local"] ]

# making line plot
X = 0:10:3000
X1 = 500:10:3000
label = ["sv" "sv_local" ]
line_plot(X,log1p.(ksd_seq .- 1.) ; folder = fig_path, name = "GMM_ksd.png", label = label, 
            xlabel = "Iteration", ylabel = "KSD", legendfontsize=18)

line_plot(X1, -log1p.(-elbo_seq[51:end,:] .- 1.);  folder = fig_path, name = "GMM_elbo.png",ylim = (-6.5,-6.25),
            xlabel = "Iteration", ylabel = "ELBO (log scale)", legendfontsize=18)

