using Random, Distributions

Random.seed!(2021);
center  = (rand(2,4) .- 0.5)*10

dat  = zeros(400, 2)
dat[1:100, :] .= randn(100, 2)*0.5 .+ center[:,1]'
dat[101:200, :] .= randn(100, 2)*0.5 .+ center[:,2]'
dat[201:300, :] .= randn(100, 2)*0.5 .+ center[:,3]'
dat[301:400, :] .= randn(100, 2)*0.5 .+ center[:,4]'

# fit with 2
save("example/GMM/data/dataset.jld", "X", Matrix(dat'))

# dat = load("example/GMM/data/dataset.jld")