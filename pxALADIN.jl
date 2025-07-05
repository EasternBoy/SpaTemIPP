"Distributed version"
function dstbProxADLADIN!(robo::Vector{robot},  Pred::Array{Float64}, NB::Vector{Vector{Int64}}, pserSet::Vector{Vector{Int64}}, 
                        mGP::Vector{GPBase}, time::Float64; MAX_ITER = 1000, thres = 1e-1)
    Dim = length(robo[1].posn)
    M   = length(robo)
    H   = robo[1].H

    NBe = [[NB[i]; i] for i in 1:M]

    ζ   = zeros(Dim, H, M, M)
    ξ   = zeros(Dim, H, M, M)
    λ   = zeros(Dim, H, M, M)
    eps = zeros(MAX_ITER, M)
    ρ   = 1e-2
    prox = 1e-3
    resl = zeros(Dim, H, M)

    Δv = zeros(M)
    Δθ = zeros(M)
    
    [ζ[:,:,i,j] = Pred[:,:,j] for i in 1:M, j in 1:M]
        
    for k in 1:MAX_ITER
        ρ    = 1.1*ρ
        prox = 1.1*prox
        
        if ρ > 10.
            ρ = 10.
        end

        if prox > 1000.
            prox = 1000.
        end

        for i in 1:M
            ξ[:,:,i,:], Δv[i], Δθ[i] = dstbProjection!(robo[i], ζ, λ, NB[i], pserSet[i], ρ, M)
        end

        for i in 1:M # Solve for p
            ζk = Matrix{Float64}(undef, Dim+1, 0)
            for j in NBe[i]
                pt = zeros(Dim+1, H)
                pt[1:Dim,:] = ζ[:,:,i,j]
                pt[Dim+1,:] = time*ones(H)
                ζk = [ζk pt]
            end
            grad     = dstbLinearize_logdet(robo[i], ζk, mGP[i], time)
            c = 0

            for j in NBe[i]
                if j == i
                    ζ[:,:,i,j] =  1/(prox + ρ*length(NBe[i]))*(sum(ρ*ξ[:,:,j,i] - λ[:,:,i,j]  for j in NBe[i]) - grad[:,(c*H+1):(c+1)*H] + prox*ζ[:,:,i,i])
                else
                    ζ[:,:,i,j] =  1/(prox + ρ)*( ρ*ξ[:,:,j,j] + prox*ζ[:,:,i,j] - grad[:,(c*H+1):(c+1)*H])
                end
                c = c+1
            end
        end
        
        for i in 1:M # Update dual variable
            for j in NBe[i]
                λ[:,:,i,j] = λ[:,:,i,j] + ρ*(ζ[:,:,i,i] - ξ[:,:,j,i])
            end
        end
        
        for i in 1:M
            eps[k,i] = sum(norm(vec(ζ[:,:,i,i] - ξ[:,:,j,i])) for j in NBe[i])
            resl[:,:,i] = ζ[:,:,i,i]
        end

        ter = maximum(eps[k,:])
        if ter < thres
            println("Terminate at $k with error = $ter")
            break
        end
    end
    return resl, eps, Δv, Δθ
end


function dstbProjection!(robo::robot, ζ::Array{Float64}, λ::Array{Float64}, NB::Vector{Int64}, 
                            pserSeti::Vector{Int64}, ρ::Float64, M::Int64)
    Dim = length(robo.posn)

    # Variables
    H = robo.H
    i = robo.index
    r = robo.r

    @variable(robo.opti, ξ[1:Dim, 1:H, 1:M])
    @variable(robo.opti, Δv[1:H])
    @variable(robo.opti, Δθ[1:H])

    # Objective function
    J = sum(dot(vec(ξ[:,:,j] - ζ[:,:,j,j] - 1/ρ*λ[:,:,j,i]), vec(ξ[:,:,j] - ζ[:,:,j,j] - 1/ρ*λ[:,:,j,i])) for j in [NB; i]) 

    @objective(robo.opti, Min, J)

    for j in pserSeti # Connectivity preserving
        for h in 1:1
            @constraint(robo.opti, dot(ξ[:,h,i] - ξ[:,h,j], ξ[:,h,i] - ξ[:,h,j]) <= robo.R^2 - 50.0) # 2.0 for residual errors of ADMM
        end
    end

    # Collision avoidance
    # for j in NB
    #     for h in 1:1
    #         e = (Pred[:,h,i] - Pred[:,h,j])/norm(Pred[:,h,i] - Pred[:,h,j])
    #         @constraint(robo.opti, dot(ξ[:,h,i] - ξ[:,h,j], e) >= 2r)
    #     end
    # end

    @constraints(robo.opti, begin robo.pBnd.x_min + r .<= ξ[1,:,i] .<= robo.pBnd.x_max - r
                                  robo.pBnd.y_min + r .<= ξ[2,:,i] .<= robo.pBnd.y_max - r
                        end)
    

    vk_1 = robo.v
    θk_1 = robo.θ
    τ    = robo.τ*10
    Ak   = I(Dim)
    Bk   = τ*[cos(θk_1) -vk_1*sin(θk_1);
              sin(θk_1)  vk_1*cos(θk_1)]
    fk   = τ*vk_1*[cos(θk_1);
                   sin(θk_1)]
    ξ0 = [robo.posn[1]; robo.posn[2]]

    for h in 1:H
        if h == 1
            @constraint(robo.opti, Ak*ξ0 + Bk*[Δv[h]; Δθ[h]] + fk .== ξ[:,h,i])    
        elseif h == 2
            @constraint(robo.opti, Ak*ξ[:,h-1,i] + Bk*[Δv[h]; Δθ[h]] + (ξ[:,h-1,i] - ξ0) .== ξ[:,h,i])
        else
            @constraint(robo.opti, Ak*ξ[:,h-1,i] + Bk*[Δv[h]; Δθ[h]] + (ξ[:,h-1,i] - ξ[:,h-2,i]) .== ξ[:,h,i])
        end
    end
    @constraints(robo.opti, begin -robo.pBnd.u_max[1] .<=  Δv .<= robo.pBnd.u_max[1] end)
    @constraints(robo.opti, begin -robo.pBnd.u_max[2] .<=  Δθ .<= robo.pBnd.u_max[2] end)

    JuMP.optimize!(robo.opti) #Solve
    ξ  = JuMP.value.(ξ)
    Δv = JuMP.value.(Δv)
    Δθ = JuMP.value.(Δθ)

    Base.empty!(robo.opti)
    return ξ, Δv[1], Δθ[1]
end