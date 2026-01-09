using MPI
using Tullio
using MAT
using LinearAlgebra

"""
Evaluates the numerical flux using Godunov's method for the Burgers' equation.

Parameters:
- ql::Array: Left state vector.
- qr::Array: Right state vector.

Returns:
- fs::Array: Numerical flux vector.
"""
function eval_numflux_godunov_burg(ql, qr)
    fl = 0.5 .* (ql .^ 2)  # Flux on the left
    fr = 0.5 .* (qr .^ 2)  # Flux on the right
    fs = zeros(size(ql))   # Final flux

    # Determine flux based on characteristic speeds
    idx0 = (ql .<= 0) .& (qr .>= 0)
    idx1 = (ql .<= qr) .& .~idx0
    idx2 = ql .> qr

    fs[idx0] .= 0
    fs[idx1] .= minimum(hcat(fl[idx1][:], fr[idx1][:]), dims=2)[:]
    fs[idx2] .= maximum(hcat(fl[idx2][:], fr[idx2][:]), dims=2)[:]

    return fs
end

"""
Splits an integer N into n parts as evenly as possible, returning an array of counts.

Parameters:
- N::Integer: Total number to split.
- n::Integer: Number of parts to split into.

Returns:
- Array: Array of counts representing the number of elements in each part.
"""
function split_count(N::Integer, n::Integer)
    q, r = divrem(N, n)
    return [i <= r ? q+1 : q for i = 1:n]
end

"""
Runs a finite volume method (FVM) simulation for the Burgers' equation using MPI parallelization.

Parameters:
- arg::Any: Argument used to name the output file.

Description:
This function initializes MPI, sets up the problem, performs the FVM simulation, and saves the results to a MAT file.
"""
function run_mpi_fvm_burgers(arg)
    # Initialize MPI
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size_mpi = MPI.Comm_size(comm)

    # Problem setup
    a = -1.0
    b = 1.0
    nelem = 4998

    x_grid = range(a, b, length=nelem+1)
    dx = x_grid[2] - x_grid[1]
    x_cc  = reshape(x_grid[1:end-1], 1,:) .+ (dx / 2)
    Q = 0.5.*(x_cc .> 0) .-0.5.*(x_cc .<= 0.0)
 

    Q0 = hcat(Q[:,1:1], Q, Q[:,end:end])
    nvar = 1
    N = nelem + 2
    CFL = 0.5
    dt = CFL * dx
    Tf = 1  # Final time
    root = 0

    nt = ceil(Int64, Tf / dt)  # Number of time steps

    # Domain decomposition
    local_N = N ÷ size_mpi
    if rank < N % size_mpi
        local_N += 1
    end

    # Local arrays
    u_local = zeros(nvar, local_N)
    u_new_local = similar(u_local)

    # Communication setup
    left = rank - 1
    right = rank + 1
    if rank == root left = MPI.PROC_NULL end
    if rank == size_mpi - 1 right = MPI.PROC_NULL end
    recv_left = Array{Float64}(undef, nvar, 1)
    recv_right = Array{Float64}(undef, nvar, 1)

    if rank == root
        output = Array{Float64}(undef, nvar, N)

        # Determine the counts for MPI.Gatherv
        M_counts = [nvar for i = 1:size_mpi]
        N_counts = split_count(N, size_mpi)
        sizes = vcat(M_counts', N_counts')
        size_ubuf = UBuffer(sizes, 2)

        counts = vec(prod(sizes, dims=1))
        output_vbuf = VBuffer(output, counts)  # VBuffer for gather
        u_full = Array{Float64}(undef, nt + 1, nvar, N)
    else
        size_ubuf = UBuffer(nothing)
        output_vbuf = VBuffer(nothing)
    end

    # Initialize the temperature distribution
    for i in 1:local_N
        global_i = rank * (N ÷ size_mpi) + i
        u_local[:, i] = Q0[:, global_i]
    end

    MPI.Gatherv!(u_local, output_vbuf, root, comm)
    if rank == root
        u_full[1, :, :] .= output
    end
    u_new_local = u_local

    # File setup for saving data
    start_time = MPI.API.MPI_Wtime()
    for step in 2:nt+1
        # Exchange data with neighbors
        send_left = (rank > 0) ? u_local[:, 1:1] : NaN
        send_right = (rank < size_mpi - 1) ? u_local[:, end:end] : NaN
        rreq_left = MPI.Irecv!(recv_left, comm; source=left)
        sreq_left  = MPI.Isend(send_left, comm; dest=left)
        rreq_right = MPI.Irecv!(recv_right, comm; source=right)
        sreq_right  = MPI.Isend(send_right, comm; dest=right)
        stats = MPI.Waitall([rreq_left, sreq_left, rreq_right, sreq_right])

        MPI.Barrier(comm)

        # Update the solution
        for i in 2:local_N-1
            u_new_local[:, i] = u_local[:, i] - dt / dx * (
                eval_numflux_godunov_burg(u_local[:, i], u_local[:, i+1]) -
                eval_numflux_godunov_burg(u_local[:, i-1], u_local[:, i])
            )
        end

        # Apply boundary conditions
        if left != MPI.PROC_NULL
            u_new_local[:, 1] = u_local[:, 1] - dt / dx * (
                eval_numflux_godunov_burg(u_local[:, 1], u_local[:, 2]) -
                eval_numflux_godunov_burg(recv_left[:, 1], u_local[:, 1])
            )
        end
        if right != MPI.PROC_NULL
            u_new_local[:, end] = u_local[:, end] - dt / dx * (
                eval_numflux_godunov_burg(u_local[:, end], recv_right[:, 1]) -
                eval_numflux_godunov_burg(u_local[:, end-1], u_local[:, end])
            )
        end

        # Swap arrays
        u_local = u_new_local

        MPI.Barrier(comm)

        # Gather the solution onto process 0
        MPI.Gatherv!(u_local, output_vbuf, root, comm)
        if rank == root
            u_full[step, :, :] .= output
        end
    end
    end_time = MPI.API.MPI_Wtime()

    if rank == root
        total_time = end_time - start_time
        println("rank $rank took $total_time seconds")
        file = matopen("burgers_$(arg)_solution.mat", "w")
        write(file, "res_u", u_full)
        close(file)
    end
    MPI.Finalize()
end

if length(ARGS) == 0
    ("Warning: No name for the output file provided using default name.")
    arg = "example"
    run_mpi_fvm_burgers(arg)
else
    arg = ARGS[1]
    run_mpi_fvm_burgers(arg)
end

