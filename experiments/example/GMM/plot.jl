#pwd() = "qua5/experiemnts"
include("example/common/result.jl")
include("example/common/util.jl")
include("example/common/plotting.jl")



result_dir = "example/GMM/result/"
fig_path = "example/GMM/figure/"

#load KSD and ELBO
K = load(joinpath(result_dir, "ksd.jld"))
E = load(joinpath(result_dir, "elbo.jld"))
ksd_seq = [ K["sv"] K["sv_local"] K["asv"] K["asv_local"] ]
elbo_seq = [ E["sv"] E["sv_local"] E["asv"] E["asv_local"] ]

# making line plot
X = 0:10:3000
X1 = 500:10:3000
label = ["sv" "sv_local" "asv" "asv_local"]
line_plot(X,log1p.(ksd_seq .- 1.) ; folder = fig_path, name = "GMM_ksd.png", label = label, 
            xlabel = "Iteration", ylabel = "log KSD ", legendfontsize=18)

line_plot(X, elbo_seq;  folder = fig_path, name = "GMM_elbo.png",label = label,
            xlabel = "Iteration", ylabel = "ELBO (log scale)", legendfontsize=18, legend = :bottomright)


line_plot(X1, log1p.(ksd_seq[51:end,:] .- 1.);  folder = fig_path, name = "GMM_ksd_zoom.png",
            xlabel = "Iteration", ylabel = "log KSD ", legendfontsize=18)

