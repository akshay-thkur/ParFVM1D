using MPI
using Tullio
using MAT
using LinearAlgebra

"""
Computes the flux vector for the shallow water equations.

Parameters:
- q::Array: State vector containing water height and momentum.
- g::Float64: Gravitational acceleration constant. Default is 1.0.

Returns:
- f::Array: Flux vector.
"""
function flux_fcn(q; g=1.0)
    h_torch = q[1:1,:]
    hu_torch = q[2:end,:]
    condition = h_torch .== 0
    f2 = ifelse.(condition, zeros(size(q[1:1,:])), (hu_torch .^ 2 ./ h_torch) .+ (g .* (h_torch .^ 2) ./ 2))
    f = vcat(hu_torch, f2)
    return f
end

"""
Computes the Riemann flux using the Roe solver for the shallow water equations.

Parameters:
- ql::Array: Left state vector.
- qr::Array: Right state vector.
- g::Float64: Gravitational acceleration constant. Default is 1.0.

Returns:
- fs::Array: Numerical flux vector.
"""
function eval_numflux_roe_swe(ql, qr, g=1.0)
    hl = ql[1:1,:]; hr = qr[1:1,:];
    ul = ql[2:2,:]./hl; ur = qr[2:2,:]./hr;
    havg = 0.5 * (hl .+ hr)
    uavg = (sqrt.(hl) .* ul .+ sqrt.(hr) .* ur) ./ (sqrt.(hl) .+ sqrt.(hr))
    cavg = sqrt.(g .* havg)
    dq = qr .- ql
    eig1 = uavg .- cavg
    eig2 = uavg .+ cavg
    cdiv = 0.5 ./ cavg
    diag_elements = vcat(eig1, eig2)
    diag_mat_abs = zeros(size(diag_elements,1),size(diag_elements,1),size(diag_elements,2))
    for i = 1:size(diag_elements,2)
        diag_mat_abs[:,:,i] = abs.(diagm(diag_elements[:,i])) 
    end

    right_eigvecs = permutedims(cat(ones(size(ql)), diag_elements,dims=3),(3,1,2))
    left_eigvecs = permutedims(cat(vcat(eig2 .* cdiv, -cdiv), vcat(-eig1 .* cdiv, cdiv),dims=3),(3,1,2))

    flux_correction = similar(ql)
    @tullio flux_correction[i] := right_eigvecs[i,j,k]*diag_mat_abs[j,l,k]*left_eigvecs[l,m,k]*dq[m]
    fs = 0.5 * (flux_fcn(ql) .+ flux_fcn(qr)) - 0.5 .* flux_correction
    return fs
end

"""
Divides an integer N into n parts.

Parameters:
- N::Integer: Total count.
- n::Integer: Number of parts.

Returns:
- Array: An array specifying the count of elements in each part.
"""
function split_count(N::Integer, n::Integer)
    q,r = divrem(N, n)
    return [i <= r ? q+1 : q for i = 1:n]
end

"""
Runs the MPI Finite Volume Method for the shallow water equations.

Parameters:
- arg: Name of the output file.
"""
function run_mpi_fvm_swe(arg)
    # Initialize MPI
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size_mpi = MPI.Comm_size(comm)

    # Problem setup
    a = 0.0
    b = 2.0
    nelem = 4998

    x_grid = range(a, b, length=nelem+1)
    dx = x_grid[2] - x_grid[1]
    x_cc  = reshape(x_grid[1:end-1], 1,:) .+ (dx / 2)
    
    Q = ifelse.(1.1 .<= x_cc .<= 1.2, [1.2,0.0], [1.0,0.0])
 
    Q0 = hcat(Q[:,1:1],Q,Q[:,end:end])
    
    nvar = 2
    N = nelem+2
    CFL = 0.1
    dt = CFL*dx
    Tf = 0.2  # Final time
    root = 0

    nt = ceil(Int64,Tf/dt) # Number of time steps
    
    # Domain decomposition
    local_N = N ÷ size_mpi
    if rank < N % size_mpi
        local_N += 1
    end
    
    # Local arrays
    u_local = zeros(nvar,local_N)
    u_new_local = similar(u_local)

    # Communication setup
    left = rank - 1
    right = rank + 1
    if rank == root left = MPI.PROC_NULL end
    if rank == size_mpi - 1 right = MPI.PROC_NULL end
    recv_left = Array{Float64}(undef,nvar,1)
    recv_right = Array{Float64}(undef, nvar,1)

    if rank == root
        output = Array{Float64}(undef,nvar,N)
        
        # Julia arrays are stored in column-major order, so we need to split along the last dimension
        M_counts = [nvar for i = 1:size_mpi]
        N_counts = split_count(N, size_mpi)
        sizes = vcat(M_counts', N_counts')
        size_ubuf = UBuffer(sizes, 2)

        counts = vec(prod(sizes, dims=1))
        output_vbuf = VBuffer(output, counts) # VBuffer for gather
        u_full = Array{Float64}(undef,nt+1,nvar,N)
    else
        size_ubuf = UBuffer(nothing)
        output_vbuf = VBuffer(nothing)
    end

    # Initialize the temperature distribution
    for i in 1:local_N
        global_i = rank * (N ÷ size_mpi) + i
        u_local[:,i] = Q0[:,global_i]
    end

    MPI.Gatherv!(u_local, output_vbuf, root, comm)
    if rank == root
        u_full[1,:,:] .= output
    end
    u_new_local = u_local
    
    # File setup for saving data
    start_time = MPI.API.MPI_Wtime()
    for step in 2:nt+1
        # Exchange data with neighbors
        send_left = (rank > 0) ? u_local[:,1:1] : NaN
        send_right = (rank < size_mpi - 1) ? u_local[:,end:end] : NaN
        rreq_left = MPI.Irecv!(recv_left, comm; source=left)
        sreq_left  =  MPI.Isend(send_left, comm; dest=left)
        rreq_right = MPI.Irecv!(recv_right, comm; source=right)
        sreq_right  =  MPI.Isend(send_right, comm; dest=right)
        stats = MPI.Waitall([rreq_left, sreq_left, rreq_right,sreq_right])

        MPI.Barrier(comm)
        # Update the solution
        for i in 2:local_N-1
            u_new_local[:,i] = u_local[:,i] -  dt / dx * (eval_numflux_roe_swe(u_local[:,i],u_local[:,i+1]) - eval_numflux_roe_swe(u_local[:,i-1],u_local[:,i]))
        end

       # Apply boundary conditions
        if left != MPI.PROC_NULL
            u_new_local[:,1] = u_local[:,1] -  dt / dx * (eval_numflux_roe_swe(u_local[:,1],u_local[:,2]) - eval_numflux_roe_swe(recv_left[:,1],u_local[:,1]))
        end
        if right != MPI.PROC_NULL
            u_new_local[:,end] = u_local[:,end] - dt / dx * (eval_numflux_roe_swe(u_local[:,end],recv_right[:,1]) - eval_numflux_roe_swe(u_local[:,end-1],u_local[:,end]))
        end

        # Swap arrays
        u_local = u_new_local

        MPI.Barrier(comm)

        # Gather the solution onto process 0
        MPI.Gatherv!(u_local, output_vbuf, root, comm)
        if rank == root
            u_full[step,:,:] .= output
        end
    end
    end_time = MPI.API.MPI_Wtime()

    if rank == root
        total_time = end_time -start_time
        println("rank $rank took $total_time seconds")
        file = matopen("swe_$(arg)_solution.mat","w")
        write(file,"res_u", u_full)
        close(file)
    end

    MPI.Finalize()
end

if length(ARGS) == 0
    println("Warning: No name for the output file provided using default name.")
    arg = "example"
    run_mpi_fvm_burgers(arg)
else
    arg = ARGS[1]
    run_mpi_fvm_burgers(arg)
end

