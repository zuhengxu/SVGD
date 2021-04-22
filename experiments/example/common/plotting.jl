using Plots, Suppressor

#generate gif 2d
function gif2d(x, y, f::Function,T::Array{Float64,3}; folder::String = "figure/", name::String = "T.gif", title::String = "")
    # create the figure folder
    if ! isdir(folder)
        mkdir(folder)
    end 

    # gif 
    anim = @suppress_err @animate for i=1:size(T, 3)
        print(i,"/",size(T, 3), "\r")
        flush(stdout)
        contourf(x, y, f,  seriescolor = cgrad(:viridis, scale = :exp), colorbar=:none,xtickfontsize=18,ytickfontsize=18, title = title)
        scatter!(T[1,:, i], T[2,:, i], alpha=0.7,label="", ylim=(y[1], y[end]), xlim=(x[1],x[end]))
    end
    gif(anim, joinpath(folder,name), fps = 15)
end


# generate contour plot
function contour_save(x, y, f::Function,T::Array{Float64,2}; folder::String = "figure/", name::String = "trace.png", title::String = "")
    # create the figure folder 
    if ! isdir(folder)
        mkdir(folder)
    end
    #contour and scatter 
    p = contourf(x, y, f,  seriescolor = cgrad(:viridis, scale = :exp), colorbar=:none,xtickfontsize=18,ytickfontsize=18, title = title)
    scatter!(T[1,:], T[2,:], alpha=0.7, label= "", ylim=(y[1], y[end]), xlim=(x[1],x[end]))
    savefig(p, joinpath(folder, name))
end

# generate gif 1d
function gif1d(x, f::Function,T::Array{Float64,2}; folder::String = "figure/", name::String = "T.gif", title::String = "")
    # create the figure folder
    if ! isdir(folder)
        mkdir(folder)
    end 

    # gif 
    anim = @suppress_err @animate for i=1:size(T, 2)
        print(i,"/",size(T, 3), "\r")
        flush(stdout)
        histogram(T[:, i], alpha=0.5,label="", normed = true, bins = 50, xlim=(x[1],x[end]))
        plot!(x, f, linewidth = 2., xtickfontsize=18,ytickfontsize=18, title = title,  label = "")
    end
    gif(anim, joinpath(folder,name), fps = 15)
end

# generate histogram
function hist_save(x, f::Function,T::Array{Float64,1}; folder::String = "figure/", name::String = "hist.png", title::String = "")
    # create the figure folder
    if ! isdir(folder)
        mkdir(folder)
    end 
    #hist 
    p = histogram(T, alpha=0.5, normed = true, label="", bins = 50,xlim=(x[1],x[end]))
    plot!(x, f, linewidth = 2., xtickfontsize=18,ytickfontsize=18, title = title, label = "")
    savefig(p, joinpath(folder, name))
end

# making line plot for quantitative comparison
function line_plot(X, T;folder::String = "figure/", name::String = "plot.png", label = "", kwargs...)
    # create the figure folder
    if ! isdir(folder)
        mkdir(folder)
    end 
    #line plot
    p = plot(X, T, lw = 3, xtickfontsize=18,ytickfontsize=18, xguidefontsize=18, yguidefontsize=18, label = label; kwargs...)
    savefig(p, joinpath(folder, name))
end

# plot(-10:.1:8, x-> 0.5*pdf(Normal(-5., 3.), x)+ 0.5*pdf(Normal(5., 0.5), x))