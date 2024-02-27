function dstbRetrain!(robo::Vector{robot}, mGP::Vector{GPBase}, NeiB::Vector{Vector{Int64}}, ts::Int64)
    M    = length(robo)
    Dim  = length(robo[1].posn) + 1

    # Hyperparameter
    for i in 1:M # Add previous data
        loca = Matrix{Float64}(undef, Dim, 0)
        obsr = Vector{Float64}(undef, 0)
        for j in NeiB[i]
            for r in 1:M
                for t in 1:ts
                    if  robo[i].data[r,t][1] == -1 && robo[j].data[r,t][1] != -1
                        robo[i].data[r,t] = robo[j].data[r,t]
                        loca = [loca  [robo[j].data[r][1:2]; t*robo[r].τ]]
                        obsr = [obsr;  robo[j].data[r][3]]
                    end
                end
            end
        end
        # append!(mGP[i], loca, obsr)
        robo[i].loca = [robo[i].loca  loca]
        robo[i].obsr = [robo[i].obsr; obsr]
    end

    for i in 1:M # Add current data
        loca = Matrix{Float64}(undef, Dim, 0)
        obsr = Vector{Float64}(undef, 0)
        for j in [NeiB[i]; i]
            robo[i].data[j,ts][1:2] = robo[j].posn
            robo[i].data[j,ts][3]   = robo[j].meas
            loca = [loca  [robo[j].posn; robo[j].time]]
            obsr = [obsr;  robo[j].meas]
        end
        # append!(mGP[i], loca, obsr)
        robo[i].loca = [robo[i].loca  loca]
        robo[i].obsr = [robo[i].obsr; obsr]
    end

    for i in 1:M
        kernel = Masked(Mat12Iso(log(mGP[i].kernel.kleft.kernel.ℓ), 1/2*log(mGP[i].kernel.kleft.kernel.σ2)), [3])*
                 Masked(SE(1/2*log(mGP[i].kernel.kright.kernel.ℓ2), 1/2*log(mGP[i].kernel.kright.kernel.σ2)), [1, 2])

        mGP[i] = GPE(robo[i].loca, robo[i].obsr, MeanConst(mGP[i].mean.β), kernel, -2.)  

        GaussianProcesses.optimize!(mGP[i], kernbounds = [[-5., -5., -5., -5.],[5., 5., 5., 5.]], noisebounds = [-3., -1.])

        if mGP[i].logNoise.value > -1. && mGP[i].logNoise.value < -3.
            mGP[i].logNoise.value = -2.
        end

        robo[i].β   = mGP[i].mean.β
        robo[i].lℓ  = mGP[i].kernel.kleft.kernel.ℓ
        robo[i].lσ2 = mGP[i].kernel.kleft.kernel.σ2

        robo[i].rℓ2 = mGP[i].kernel.kright.kernel.ℓ2
        robo[i].rσ2 = mGP[i].kernel.kright.kernel.σ2

        robo[i].σω2 = exp(2*mGP[i].logNoise.value)

        robo[i].iCθ = inv(dstbSEkernel(robo[i], robo[i].loca, robo[i].loca) + robo[i].σω2*I(length(robo[i].obsr)))
    end
end



function myGP_predict(mGP::GPBase, p::Matrix{Float64}, full_cov = true)
    return predict_y(mGP, p, full_cov = full_cov)
end


function dstbLinearize_logdet(robo::robot, u::Matrix{Float64}, mGP::GPBase)
    numDa     = length(robo.obsr)
    Cuu       = dstbSEkernel(robo, u, u)
    Coo       = robo.iCθ
    Cou       = dstbSEkernel(robo, robo.loca, u)
    Cuo       = Matrix(Cou')    
    iCθ       = inv(myGP_predict(mGP, u, true)[2])

    Dim, H    = size(u)
    Dim       = Dim - 1
    ∇L        = zeros(Dim, H)

    for r in 1:Dim
        for h in 1:H
            Ω1 = zeros(H, H)
            Ω2 = zeros(H, H)

            for k in 1:H
                Ω1[h,k] = Ω1[k,h] = -Cuu[h,k]/robo.rℓ2*(u[r,h] - u[r,k])
            end

            dCuo = zeros(H, numDa)
            for k in 1:numDa
                dCuo[h,k] = -Cuo[h,k]/robo.rℓ2*(u[r,h] - robo.loca[r,k])
            end

            Ω2 = 2*dCuo*Coo*Cou           
            ∇L[r,h] = -tr(iCθ*(Ω1 - Ω2))
        end
    end
    return ∇L
end


function dstbSEkernel(robo::robot, x::Matrix{Float64}, y::Matrix{Float64})
    row   = size(x)[2]
    col   = size(y)[2]
    C     = zeros(row, col)

    for i in 1:row
        for j in 1:col
            C[i,j] = robo.lσ2*exp(-1/2*abs(x[3,i] - y[3,j])/robo.lℓ)*
                     robo.rσ2*exp(-1/2*dot(x[1:2,i] - y[1:2,j], 1/robo.rℓ2, x[1:2,i] - y[1:2,j]))
        end
    end

    return C
end


"Take measurement"
function measure!(posn::Vector{Float64}, mGP::GPBase)
    return predict_y(mGP, posn[:,:])[1][1]
end

function meshgrid(x, y)
    # Function to make a meshgrid
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    X = reshape(X, 1, length(X))
    Y = reshape(Y, 1, length(Y))
    return [X; Y]
end
