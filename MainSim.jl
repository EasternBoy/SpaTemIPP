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
include("pxadmm.jl")


ENV["GKSwstype"]="nul"

M = 6; L= 50; 
H = 3; τ = 0.1; 
MAX_ITER = 100
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

numData = length(ID)
inDim   = 3

scale  = 10
scaNum = Int64(round(numData/scale))-1
time   = [readTime(timestr[i*scale]) for i in 1:scaNum]

# arrC    = [[] for i in 1:size(coor)[1]]
# arrTime = [[] for i in 1:size(coor)[1]]
# for i in 1:numData
#     append!(arrC[ID[i]], data[i])
#     append!(arrTime[ID[i]], time[i])
# end
# for i in 1:12
#     Fig1 = plot(arrTime[i],arrC[i])
#     png(Fig1, "Sensor$i")
# end


spaTem  = zeros(inDim,scaNum)
gpData  = zeros(scaNum)

for i in 1:scaNum
    spaTem[:,i] = [coor[ID[i*scale],:]; time[i]]
    gpData[i] = data[i*scale]
end


kernel = Masked(Mat12Iso(1., 0.), [3])*Masked(SE(1., 0.), [1, 2])
# kernel = SEArd([0., 0., 0.], 0.)
mean   = MeanConst(Statistics.mean(gpData))

# Create a ground-truth model from the data
# GPtruth = GPE(spaTem, gpData, mean, kernel, -2.)

# println("ExactGP Training time")
# @time GaussianProcesses.optimize!(GPtruth, noisebounds = [-3., -1.])





testSize = [50, 50]
timeLen  = 24
Period   = 1
timeScale = 0:Period:timeLen
PreMat = [zeros(3, testSize[1]*testSize[2]) for t in timeScale]
X   = range(0,   stop = 100, length = testSize[1])
Y   = range(-15, stop = 5,   length = testSize[2])
for (index,t) in enumerate(timeScale)
    count = 1
    for i in X, j in Y
        PreMat[index][:,count] = [i,j,t]
        count = count+1
    end
end
for (index,t) in enumerate(timeScale)
    vectemp  = myGP_predict(GPtruth, PreMat[index])[1] #Take only mean
    temp     = reshape(vectemp, testSize[1], testSize[2])'
    Fig0 = heatmap(X, Y,  temp, c = :turbo, tickfontsize = 14, xlims = (x_min,x_max), ylims = (y_min,y_max), 
                    size=(600,300), clims = (minimum(GPtruth.y), maximum(GPtruth.y)), rightmargin=5Plots.mm)
    png(Fig0, "GroudTruth at $t h")
end


s_max = 15.; R = 15.; r = 1.
pBounds  = polyBound(s_max, x_min, x_max, y_min, y_max)

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
minD         = ones(L)*1000000
ResE         = ones(MAX_ITER, M, L)
PosX         = zeros(M,L)
PosY         = zeros(M,L)
[Pred[:,h,i] = robo[i].posn for h in 1:H, i in 1:M]

var     = Vector{Vector{Float64}}(undef,L)
RMSE    = zeros(L)

for k in 1:L
    println("Time instance $k")
    global Pred, ResE, NB

    # Train
    @time dstbRetrain!(robo, mGP, NB, k)

    NB      = find_nears(robo, M)
    Eig2[k] = Index!(NB)
    pserSet = pserCon(robo)

    Fig, RMSE[k], var[k] = myPlot(robo, mGP, GPtruth, k*τ, NB, color)
    png(Fig, "Fig-5robots/step $k"); display(Fig)

    # Execute PxADMM
    @time Pred, ResE[:,:,k] = dstbProxADLADIN!(robo, Pred, NB, pserSet, mGP, k*τ; MAX_ITER = MAX_ITER)

    # Robots move to new locations and take measurement
    for i in 1:M
        robo[i].posn = Pred[:,1,i]
        robo[i].time = k*τ
        robo[i].meas = measure!([robo[i].posn; robo[i].time], GPtruth)
    end
end

# plot(collect(1:L),var)