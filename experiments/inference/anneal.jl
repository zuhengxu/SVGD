# Different annealing sched

# cyclical annealing sched
@inline function cyclical_sched(i, npercycle::Int64, p::Float64)
    return (mod(i, npercycle)/(npercycle))^p
end


# no annealing (i -> 1.)
id_sched = i::Int64 -> 1.



