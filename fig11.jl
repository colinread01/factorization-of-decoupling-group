

using QuantumOptics
using Makie 
using CairoMakie 
using GLMakie 
using Colors
using JLD
using LinearAlgebra
GLMakie.activate!()

#################
# Generate Data #
#################

# generators in axis-angle notation 

g1 = ([sqrt(2/3) , 0  ,sqrt(1/3)], 1.0*pi)
g2 = ([-sqrt(1/6) , sqrt(1/2) , sqrt(1/3)], 1.0*pi)
g3 = ([0,0,1],2*pi/3)
g4 = ([-sqrt(1/3) , 0  ,sqrt(2/3)] , 1.0*pi)
g5 = ([0,0,1],2*pi/3)
g6 = ([sqrt(2)/3 , sqrt(2/3) , 1/3], 2*pi/3)
mg5 = (-[0,0,1],2*pi/3)
mg6 = (-[sqrt(2)/3 , sqrt(2/3) , 1/3], 2*pi/3)
g8 = ([-sqrt(1/6), sqrt(1/2), sqrt(1/3)], 1.0*pi)

# sequences 

Ta = [g5 , g6 , g5 , g5 , mg6 , mg5 , mg5 ,mg6 , mg6 , g5 , g5 , mg6]
Tb = [g8 , g5 , g5 , g8 , mg5 , mg5 , g8 , g5 , g5 , g8 , mg5 , mg5]
C3D2 = [[g1] , [g2] , [g1] , [g2, g3] , [g1] , [g2] , [g1] , [g2, g3],[g1] , [g2] , [g1] , [g2, g3]]


function rotation(b, axis, angle) # rotation pulse 

    J = [sigmax(b)/2 , sigmay(b)/2 , sigmaz(b)/2]

    return exp(dense(-im*angle*sum(axis[n]*J[n] for n in 1:3)))
end

function Tlmdk(N::Int, L::Int, M::Int) # Spherical tensors 


    Tlm = Array{Complex{Float64},2}(undef, N+1, N+1)

    for k in 0:N 
        for kp in 0:N 
            if N/2-k in -N/2:N/2 && N/2-kp in -N/2:N/2
                Tlm[k+1,kp+1] = sqrt((2*L+1)/(N+1))*clebschgordan(N/2,N/2-kp,L,M,N/2,N/2-k)
            else 
                Tlm[k+1,kp+1] = 0
            end
        end
    end

    return Tlm
end

function propagate_seq(U0,seq, b) # propagate sequence (for the sequence C3D2)

    # U0 = free-evolution propagator 
    U = exp(-im*0.0*one(b))

    for n in 1:length(seq)
        U = U0*U 
        for m in 1:length(seq[n])
            U = rotation(b, seq[n][m][1], seq[n][m][2])*U
        end
    end
    
    return log10(sqrt(abs(1 - (1/tr(one(b)))*abs(tr(U)))))
end

function propagate_seq2(U0,seq, b) # propagate sequence (for TDD1 and TDD2)

    U = exp(-im*0.0*one(b))

    for n in 1:length(seq)

        U = U0*U 
        U = rotation(b, seq[n][1], seq[n][2])*U

    end
    
    return log10(sqrt(abs(1 - (1/tr(one(b)))*abs(tr(U)))))
end

function linear_noise(b) # noise Hamiltonian

    J = [sigmax(b)/2 , sigmay(b)/2 , sigmaz(b)/2]
    ax = [rand()-0.5 , rand()-0.5 , rand()-0.5]
    ax = ax/norm(ax)
    Hnoise = sum(ax[n]*J[n] for n in 1:3)

    return Hnoise/norm(Hnoise,Inf)
end


data_dir = "/home/colin/Documents/JULIA/Codes persos/Platonic_sequences/Multisymmetrization/Codes 4 GitHub/Data_fig11/"

b = SpinBasis(3.0)

lognoise = LinRange(-1.8,0.2,200)
noise = exp10.(lognoise)

nsample = 5000

data0 = Vector{Float64}(undef, length(noise))
data_Ta = Vector{Float64}(undef, length(noise))
data_Tb = Vector{Float64}(undef, length(noise))
data_Hierarchy = Vector{Float64}(undef, length(noise))

Threads.@threads for i in 1:length(noise)

    moy0 = Vector{Float64}(undef, nsample)
    moy_Ta = Vector{Float64}(undef, nsample)
    moy_Tb = Vector{Float64}(undef, nsample)
    moy_H = Vector{Float64}(undef, nsample)

     for n in 1:nsample 

        hn = linear_noise(b)
        U0 = exp(dense(-im*noise[i]*hn))

        moy0[n] = log10(sqrt(abs(1 - (1/tr(one(b)))*abs(tr(U0)))))
        moy_Ta[n] = propagate_seq2(U0, Ta,b)
        moy_Tb[n] = propagate_seq2(U0, Tb,b)
        moy_H[n] = propagate_seq(U0, C3D2,b)

    end
    moy_H_clean = []
    for n in 1:nsample 
        if abs(moy_H[n]) != Inf 
            push!(moy_H_clean, moy_H[n])
        end
    end


    data0[i] = sum(moy0[n] for n in 1:nsample)/nsample 
    data_Ta[i] =sum(moy_Ta[n] for n in 1:nsample)/nsample 
    data_Tb[i] =sum(moy_Tb[n] for n in 1:nsample)/nsample 
    data_Hierarchy[i] = sum(moy_H_clean[n] for n in 1:length(moy_H_clean))/length(moy_H_clean) 

    println("$i/100")
end

jldopen(data_dir*"TDDa.jld", "w") do file 
    write(file, "data" ,data_Ta)
end
jldopen(data_dir*"TDDb.jld", "w") do file 
    write(file, "data" ,data_Tb)
end
jldopen(data_dir*"C3D2.jld", "w") do file 
    write(file, "data" ,data_Hierarchy)
end
jldopen(data_dir*"NoDD.jld", "w") do file 
    write(file, "data" ,data0)
end

#############
# Load Data #
#############

data_Ta = load(data_dir*"TDDa.jld", "data")

data_Tb = load(data_dir*"TDDb.jld", "data")

data_Hierarchy = load(data_dir*"C3D2.jld", "data")

data0 = load(data_dir*"NoDD.jld", "data")

lognoise = LinRange(-1.8,0.2,200)

#############
# Plot Data #
#############

labsize = 25.0f0
ticksize = 20.0f0

fig = Figure(;size=(450,300))

ax = Axis(fig[1,1], xlabel=L"\tau \gamma " , ylabel = L"\bar{D}",
 ylabelsize = labsize, xlabelsize=labsize ,
 xticks = ([-1,0],[L"10^{-1}", L"1"]), yticks = ([-4,-3,-2,-1,0], [L"10^{-4}",L"10^{-3}",L"10^{-2}", L"10^{-1}",L"1"]),
 xminorticks = [log10(0.9), log10(0.8),log10(0.7),log10(0.6),log10(0.5),log10(0.4),log10(0.3),log10(0.2)],
  xticklabelsize=ticksize, yticklabelsize=ticksize,ylabelrotation=0 ,xminorticksvisible = false, xminorgridvisible = false)

lines!(lognoise, data0,color=:blue, linestyle=(:dash), linewidth = 3.0, label=L"\text{NoDD}")
lines!(lognoise, data_Ta, color=:red, linestyle=(:dashdot, 3), linewidth = 3.0, label=L"\text{TDD_1}")
lines!(lognoise, data_Tb, color=:green, linestyle=(:dot, 1), linewidth = 3.0, label=L"\text{TDD_2}")
lines!(lognoise, data_Hierarchy, color=:black, linestyle=:solid , linewidth = 3.0, label=L"\text{C$_3$[D$_2$]}")

axislegend(ax,position=:rb, labelsize = 20.0f0, patchsize=(30,20))
display(fig)

CairoMakie.activate!()
save(data_dir*"fig_10.pdf", fig)