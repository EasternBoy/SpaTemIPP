push!(LOAD_PATH, ".")

import Pkg
using Pkg
Pkg.activate(@__DIR__)

using Optim, Random, Distributions, CSV, DataFrames, MAT, JLD2
using Plots, Dates,  Statistics, Colors, ColorSchemes, StatsPlots
using SharedArrays,  Ipopt, JuMP, GaussianProcesses, LinearAlgebra, Optim

include("robots.jl")
include("connectivity.jl")
include("Tools.jl")
include("computing.jl")
include("pxALADIN.jl")


ENV["GKSwstype"]="nul"

M = 6; L = 80; 
H = 2; τ = 0.1; 
MAX_ITER = 1000
x_min = 0.;   x_max = 100.
y_min = -15.; y_max = 5.
color = cgrad(:turbo, M, categorical = true, scale = :lin)


## Load data

coorfr         = CSV.read("Manzano2022COMPAG_coordinates.csv", DataFrame, header = 1; select=[2,3])
timefr         = CSV.read("Manzano2022COMPAG_data.csv", DataFrame, header = 1, select=[1])
IDfr           = CSV.read("Manzano2022COMPAG_data.csv", DataFrame, header = 1; select=[2])
datafr         = CSV.read("Manzano2022COMPAG_data.csv", DataFrame, header = 1, select=[3])

coor    = Matrix{Float64}(coorfr)
timestr = Matrix{String}(timefr)
data    = Matrix{Float64}(datafr)
ID      = Matrix{Int64}(IDfr)

inDim   = 3

numData = Int64(round(length(ID)))
time    = [readTime(timestr[i]) for i in 1:numData]
time    = time .- time[1]

arrC    = [[] for i in 1:size(coor)[1]]
arrTime = [[] for i in 1:size(coor)[1]]
for i in 1:numData
    append!(arrC[ID[i]], data[i])
    append!(arrTime[ID[i]], time[i])
end
Fig1 = plot(arrTime[10]*10,arrC[10], size=(1200,500), tickfontsize = 18, legendfontsize = 18,  linewidth=3, label = "Temperature (in Celsius scale)")
png(Fig1, "Sensor 10")

# spaTem  = zeros(inDim,scaNum)
# gpData  = zeros(scaNum)
# for i in 1:scaNum
#     spaTem[:,i] = [coor[ID[i*scale],:]; time[i]]
#     gpData[i] = data[i*scale]
# end

spaTem  = zeros(inDim,numData)
gpData  = zeros(numData)
for i in 1:numData
    spaTem[:,i] = [coor[ID[i],:]; time[i]]
    gpData[i] = data[i]
end


kernel = Masked(Mat12Iso(1., 0.), [3])*Masked(SEArd([1., 1.], 0.), [1, 2])
mean   = MeanConst(Statistics.mean(gpData))
GPtruth = GPE(spaTem, gpData, mean, kernel, -2.)
@time GaussianProcesses.optimize!(GPtruth)





testsizeX = 70
testsizeY = 35
timeLen   = 10 #hours
Period    = 0.1
timeScale = 0:Period:timeLen
PreMat = [zeros(3, testsizeX*testsizeY) for t in timeScale]
X   = Vector(range(0,   stop = 100, length = testsizeX))
Y   = Vector(range(-15, stop = 5, length = testsizeY))
for (index,t) in enumerate(timeScale)
    count = 1
    for i in X
        for j in Y
            PreMat[index][:,count] = [i,j,t]
            count = count+1
        end
    end
end
for (index,t) in enumerate(timeScale)
    vectemp  = myGP_predict(GPtruth, PreMat[index])[1] #Take only mean
    temp     = reshape(vectemp,(testsizeY,testsizeX))
    Fig0     = heatmap(X, Y,  temp, c = :turbo, tickfontsize = 14, xlims = (x_min,x_max), ylims = (y_min,y_max), 
                        size=(600,250), clims = (minimum(GPtruth.y)-2, maximum(GPtruth.y)), rightmargin=5Plots.mm)
    step = Int(round(t/Period))
    step > 9 ? png(Fig0, "Figs/GTstep$step") : png(Fig0, "Figs/GTstep0$step")
end


u_max = [2., 1.]; R = 25.; r = 1.
pBounds  = polyBound(u_max, x_min, x_max, y_min, y_max)

init    = init_position(pBounds, R, r, M)
robo    = [robot(i, τ, H, R, r, 0., pBounds, init[:,i]) for i in 1:M]
NB      = find_nears(robo, M) 
mGP     = Vector{GPBase}(undef, M)

for i in 1:M
    robo[i].meas   = measure!([robo[i].posn; robo[i].time], GPtruth)  
    mGP[i]         = GPE([1. 2.; 1. 2.; 0. 0.], [28.; 26.], MeanConst(27.), Masked(Mat12Iso(1., 0.), [3])*Masked(SE(1., 0.), [1, 2]), -2.)
end


println("Now start the simulation")
timer        = zeros(L)
Pred         = zeros(inDim-1, H, M)
Eig2         = zeros(L)
ResE         = ones(MAX_ITER, M, L)
Traj         = zeros(inDim-1, M, L)
[Pred[:,h,i] = robo[i].posn for h in 1:H, i in 1:M]

var     = Vector{Vector{Float64}}(undef,L)
RMSE    = zeros(L)

Δv = zeros(M,L)
Δθ = zeros(M,L)

for k in 1:L
    println("Time instance $k")
    global Pred, ResE, NB

    # Train
    @time dstbRetrain!(robo, mGP, NB, k)

    NB      = find_nears(robo, M)
    Eig2[k] = Index!(NB)
    pserSet = pserCon(robo)

    Fig, RMSE[k], var[k] = myPlot(robo, mGP, GPtruth, k*τ, NB, color)
    k > 9 ? png(Fig, "Figs/step$k") : png(Fig, "Figs/step0$k")

    # Execute PxADMM
    @time Pred, ResE[:,:,k], Δv[:,k], Δθ[:,k] = dstbProxADLADIN!(robo, Pred, NB, pserSet, mGP, k*τ; MAX_ITER = MAX_ITER)

    # Robots move to new locations and take measurement
    for i in 1:M
        robo[i].posn = Traj[:,i,k] =  Pred[:,1,i]
        robo[i].time = k*τ
        robo[i].v    = robo[i].v + Δv[i,k]
        robo[i].θ    = robo[i].θ + Δθ[i,k]
        robo[i].meas = measure!([robo[i].posn; robo[i].time], GPtruth)
    end
end

g = boxplot()
for i in [1;Vector(4:4:80)]
    g = boxplot!(["$i"],var[i],legend = false,size=(1100,450), xticks = :all, tickfontsize = 14)
end
display(g)
png(g, "var_steps")