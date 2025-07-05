mutable struct polyBound
    u_max::Vector{Float64} # constant of bound constraint
    x_min::Float64 
    x_max::Float64 # constant of bound constraint
    y_min::Float64 
    y_max::Float64 # constant of bound constraint

    function polyBound(u_max::Vector{Float64}, x_min::Float64, x_max::Float64, y_min::Float64, y_max::Float64)
        return new(u_max, x_min, x_max, y_min, y_max)
    end
end

mutable struct robot
    index::Int64
    τ::Float64 # sampling time
    H::Integer # Horizon length
    R::Float64
    r::Float64
    σ::Float64  # Gaussian Noise on the measurement
    pBnd::polyBound # Interval Bounds

    loca::Matrix{Float64}  # All locations up to current time
    obsr::Vector{Float64}  # All observation up to current time
    data::Matrix{Vector{Float64}}

    posn::Vector{Float64}  # Current states: x, y
    time::Float64
    meas::Float64 # measurement

    # odeprob # ODEProblem for solving the ODE
    opti

    β::Float64

    lℓ::Float64
    lσ2::Float64

    rℓ2::Vector{Float64}
    rσ2::Float64

    σω2::Float64
    iCθ::Matrix{Float64}

    θ::Float64
    v::Float64



    function robot(index::Int64, τ::Float64, H::Integer, R::Float64, r::Float64, σ::Float64, pBnd::polyBound, x0::AbstractVector)

        obj         = new(index, τ, H, R, r, σ, pBnd)
        obj.posn    = x0  # Copy x0
        obj.time    = 0.

        obj.θ = rand(1)[1]
        obj.v = 0.

        obj.data    = [-ones(3) for i in 1:M, j in 1:L+1] # -1 means no data
        obj.loca    = Matrix{Float64}(undef, length(x0)+1, 0)
        obj.obsr    = Vector{Float64}(undef, 0)

        obj.opti = JuMP.Model(Ipopt.Optimizer)
        set_silent(obj.opti)
        return obj
    end
end