

#packages

using DifferentialEquations
using QuantumOptics 
using CairoMakie
using GLMakie 
using JLD

include("/home/colin/Documents/JULIA/Codes externes/OQLiege_base.jl") # Implementation of HOPS 
include("/home/colin/Documents/JULIA/Codes externes/OQLiege.jl") # Implementation of HOPS 

data_dir = "" # data directory 

#initial state
jj = 2.0
N = floor(Int, 2*jj)
b = SpinBasis(jj)
Js = [sigmax(b) , sigmay(b) , sigmaz(b)]/2

c1 = (-1/sqrt(2)+im)/sqrt(6)
c2 = (sqrt(2)+im)/sqrt(6)
ψ0 = Ket(b,[c1 , 0 , c2 , 0 , c1])

#coupling parameters
g = 1.0 
ωc = 1.0
κ = 0.1 
ω0 = 0.0
H = ω0*dense(sigmaz(b)).data
L = Tlmdk(N, 1,0) + Tlmdk(N , 2 , 0) + Tlmdk(N , 3 , 0) + Tlmdk(N , 4 , 0)

#simulation parameters
tf = 4.0
nsteps = 8000
ntraj = 500 
kmax = 6 
ts = LinRange(0.0,tf , nsteps+1)
dt = tf/nsteps 

ψ0_hops = vcat(ψ0.data , [0.0 + im*0.0 for n in 1:length(ψ0)*kmax])
op = ψ0.data*ψ0.data'


#################
# Generate Data # 
#################

# No DD 

ρ = computeHOPS_t((0.0,tf) , dt , g , im*ωc + κ , N , ntraj , t -> H , L , op , kmax , ψ0_hops , loadingBar = true,computeρ = true, computeSTD = true) 
trρ = [real(tr(op^2)) for op in ρ[4]]

jldopen(data_dir * " HOPS NoDD ωc=$ωc κ=$κ g=$g - tf=$tf.jld", "w") do file
    write(file, "data", (trρ,ρ[3]))
end

# DD 

function H_dd(t, seq , τ , τps, χ)
    
    # τ = waiting time 
    # τps = duration of each pulse of the sequence 
    # χ = amplitude of the pulse 

    T = sum(τ + τps[n] for n in 1:length(seq))

    Js = [sigmax(b) , sigmay(b) , sigmaz(b)]/2

    t_mod = t%T

    for n in 1:length(seq)

        if n > 1 

            if (t_mod >= (sum(τ + τps[m] for m in 1:n-1) + τ)) && (t_mod <= sum(τ + τps[m] for m in 1:n))

                return χ*dense(sum(seq[n][1][m]*Js[m] for m in 1:3)).data

            end

        else 

            if t_mod >= 0.0 + τ && t_mod <= τ + τps[1]

                return χ*dense(sum(seq[n][1][m]*Js[m] for m in 1:3)).data

            end

        end

    end

    return χ*dense(one(b)).data

end

function RotationArbitrary(n, θ)

    # n = axis of rotation 
    # θ = angle of rotation 

    M = Matrix{Float64}(undef, 3,3)
    M[1,1] = cos(θ) + n[1]^2*(1-cos(θ))
    M[2,2] = cos(θ) + n[2]^2*(1-cos(θ))
    M[3,3] = cos(θ) + n[3]^2*(1-cos(θ))

    M[1,2] = n[1]*n[2]*(1-cos(θ)) - n[3]*sin(θ) 
    M[1,3] = n[1]*n[3]*(1-cos(θ)) + n[2]*sin(θ) 

    M[2,1] = n[1]*n[2]*(1-cos(θ)) + n[3]*sin(θ) 
    M[2,3] = n[3]*n[2]*(1-cos(θ)) - n[1]*sin(θ) 

    M[3,1] = n[1]*n[3]*(1-cos(θ)) - n[2]*sin(θ) 
    M[3,2] = n[3]*n[2]*(1-cos(θ)) + n[1]*sin(θ) 

    return M
end

ϕ = (1+sqrt(5))/2 # golden ratio 
gen1 = RotationArbitrary([0,0,1],-4*pi/5)*RotationArbitrary([1,0,0],-acos(ϕ/sqrt(ϕ+2)))*[1-ϕ,0,ϕ]/sqrt(3) # axis of rotation 1 
gen2 = RotationArbitrary([1,0,0],-acos(ϕ/sqrt(ϕ+2)))*[0,0,1] # axis of rotation 2
gen3 = RotationArbitrary(gen2,pi/2)*gen1 # axis of rotation 3

# sequence parameters 

ωcsτ = [0.3,0.15,0.075] # cut-off frequency X pulse interval 

for ωcτ in ωcsτ

    γ = 0.8 # ratio between the waiting time and the pulse duration for a pulse of angle π
    τ = γ*ωcτ/ωc # waiting time 
    τπ = (1-γ)*ωcτ/ωc # duration of π pulse 
    τ2π3 = (2/3)*τπ # duration of 2π/3 pulse 
    χ = pi/τπ # pulse amplitude 

    # TDD1
    R1 = (gen1, 2*pi/3)
    R2 = (gen3, 2*pi/3)
    R1m = (-gen1, 2*pi/3)
    R2m = (-gen3, 2*pi/3)
    seq1 = [ R1 , R2 , R1 , R1 , R2m , R1m , R1m , R2m , R2m , R1 , R1 , R2m]
    τ1 = [τ2π3 for n in 1:12]

    # TDD2
    R3 = (gen2, pi)
    seq2 = [ R3 , R1 , R1 , R3 , R1m , R1m ,  R3 , R1 , R1 , R3  , R1m , R1m ]
    τ2 = [ τπ , τ2π3 , τ2π3 , τπ , τ2π3 , τ2π3 , τπ , τ2π3 , τ2π3 , τπ , τ2π3 , τ2π3]

    ρ_dd1 = computeHOPS_t((0.0,tf) , dt , g , im*ωc + κ , N , ntraj , t -> H_dd(t,seq1,τ,τ1,χ) , L , op , kmax , ψ0_hops , loadingBar = true,computeρ = true , computeSTD = true  )
    ρ_dd2 = computeHOPS_t((0.0,tf) , dt , g , im*ωc + κ , N , ntraj , t -> H_dd(t,seq2,τ,τ2,χ) , L , op , kmax , ψ0_hops , loadingBar = true,computeρ = true, computeSTD = true  )

    trρ_dd1 = [real(tr(op^2)) for op in ρ_dd1[4]]
    trρ_dd2 = [real(tr(op^2)) for op in ρ_dd2[4]]

    jldopen(data_dir * " HOPS Ta γ=$γ ωcτ=$ωcτ - ωc=$ωc κ=$κ g=$g - tf=$tf.jld", "w") do file
        write(file, "data", (trρ_dd1,ρ_dd1[3]))
    end

    jldopen(data_dir * " HOPS Tb γ=$γ ωcτ=$ωcτ - ωc=$ωc κ=$κ g=$g - tf=$tf.jld", "w") do file
        write(file, "data", (trρ_dd2,ρ_dd2[3]))
    end

end


#############
# Load Data #
#############

# noDD 
trρ, Δρ = load(data_dir * " HOPS NoDD ωc=$ωc κ=$κ g=$g - tf=$tf.jld", "data")
dt = tf/nsteps 

# DD ωcτ = 0.15 γ = 0.9
γt = 0.8
trρ_dd1_b, Δρ_dd1_b  = load(data_dir * " HOPS Ta γ=$(γt) ωcτ=$(0.15) - ωc=$ωc κ=$κ g=$g - tf=$tf.jld", "data")
trρ_dd2_b, Δρ_dd2_b  = load(data_dir * " HOPS Tb γ=$(γt) ωcτ=$(0.15) - ωc=$ωc κ=$κ g=$g - tf=$tf.jld", "data")
T_dd1_b = 12*γt*0.15/ωc + 12*(1-γt)*0.15*(2/3)/ωc
T_dd2_b = 12*γt*0.15/ωc + 8*(1-γt)*0.15*(2/3)/ωc + 4 *(1-γt)*0.15/ωc 

ind1_b = []
for m in 1:length(ts)
    if (ts[m]/T_dd1_b)%1 > 1 - dt 
        push!(ind1_b, m)
    end
end
ind1_b
ind2_b = []
for m in 1:length(ts)
    if (ts[m]/T_dd2_b)%1 > 1 - dt
        push!(ind2_b, m)
    end
end
ind2_b

# DD ωcτ = 0.075 γ = 0.8
γt = 0.8
trρ_dd1_c, Δρ_dd1_c  = load(data_dir * " HOPS Ta γ=$(γt) ωcτ=$(0.075) - ωc=$ωc κ=$κ g=$g - tf=$tf.jld", "data")
trρ_dd2_c, Δρ_dd2_c  = load(data_dir * " HOPS Tb γ=$(γt) ωcτ=$(0.075) - ωc=$ωc κ=$κ g=$g - tf=$tf.jld", "data")
T_dd1_c = 12*γt*0.075/ωc + 12*(1-γt)*0.075*(2/3)/ωc
T_dd2_c = 12*γt*0.075/ωc + 8*(1-γt)*0.075*(2/3)/ωc + 4 *(1-γt)*0.075/ωc 

ind1_c = []
for m in 1:length(ts)
    if (ts[m]/T_dd1_c)%1 > 1 - 2*dt 
        push!(ind1_c, m)
    end
end
ind1_c

ind2_c = []
for m in 1:length(ts)
    if (ts[m]/T_dd2_c)%1 > 1 - 2*dt 
        push!(ind2_c, m)
    end
end
ind2_c

# DD ωcτ = 0.3 γ = 0.8
γt = 0.8
trρ_dd1_d, Δρ_dd1_d = load(data_dir * " HOPS Ta γ=$(γt) ωcτ=$(0.3) - ωc=$ωc κ=$κ g=$g - tf=$tf.jld", "data")
trρ_dd2_d, Δρ_dd2_d  = load(data_dir * " HOPS Tb γ=$(γt) ωcτ=$(0.3) - ωc=$ωc κ=$κ g=$g - tf=$tf.jld", "data")
T_dd1_d= 12*γt*0.3/ωc + 12*(1-γt)*0.3*(2/3)/ωc
T_dd2_d = 12*γt*0.3/ωc + 8*(1-γt)*0.3*(2/3)/ωc + 4*(1-γt)*0.3/ωc 

ind1_d = []
for m in 1:length(ts)
    if (ts[m]/T_dd1_d)%1 > 1 - dt/4
        push!(ind1_d, m)
    end
end
ind1_d
ind2_d = []
for m in 1:length(ts)
    if (ts[m]/T_dd2_d)%1 > 1 - dt/2
        push!(ind2_d, m)
    end
end
ind2_d


#############
# Plot Data #
#############

marks = [ MarkerElement( marker=:diamond,color=(:black,0.5), strokecolor=:black,strokewidth = 2, markersize=12) ,
 MarkerElement( marker=:rtriangle,color=(:black,0.5), strokecolor=:black,strokewidth = 2, markersize=12) ,
 MarkerElement( marker=:circ,color=(:black,0.5), strokecolor=:black,strokewidth = 2, markersize=12)]

fig = Figure(;size=(600,300))
ax = Axis(fig[1,1],ylabel = L"\mathcal{P}", xlabel = L"\omega_c t",
 ylabelrotation=0.0, xlabelsize=25.0f0,ylabelsize=25.0f0, 
 title=L"\mathrm{TDD_1}", titlesize=20.0f0,
 xticks = ([0,1,2,3,4],[L"0",L"1",L"2",L"3",L"4"]), yticks = ([0.4,0.5,0.6,0.7,0.8,0.9,1.0],[L"0.4",L"0.5",L"0.6",L"0.7",L"0.8",L"0.9",L"1.0"]))

 lines!(ts , trρ , color=:blue )
band!(ts , trρ - Δρ , trρ + Δρ, color = (:blue, 0.2))


lines!(ts , trρ_dd1_c , color=:red ,linestyle=:solid)
scatter!([ts[m] for m in ind1_c] , [trρ_dd1_c[m] for m in ind1_c], marker=:diamond,color=(:red,0.2), strokecolor=:red,strokewidth = 1.0, markersize=9)
band!(ts , trρ_dd1_c - Δρ_dd1_c , trρ_dd1_c + Δρ_dd1_c, color = (:red, 0.2))

lines!(ts , trρ_dd1_b , color=(:red,0.5) ,linestyle=:solid )
scatter!([ts[m] for m in ind1_b] , [trρ_dd1_b[m] for m in ind1_b], marker=:rtriangle,color=(:red,0.2), strokecolor=:red,strokewidth = 1.0, markersize=9)
band!(ts , trρ_dd1_b - Δρ_dd1_b , trρ_dd1_b + Δρ_dd1_b, color = (:red, 0.2))

lines!(ts , trρ_dd1_d , color=(:red,0.2)  ,linestyle=:solid )
scatter!([ts[m] for m in ind1_d] , [trρ_dd1_d[m] for m in ind1_d], marker=:circ,color=(:red,0.5), strokecolor=:red,strokewidth = 1.0, markersize=9)
band!(ts , trρ_dd1_d - Δρ_dd1_d , trρ_dd1_d + Δρ_dd1_d, color = (:red, 0.2))

ylims!(0.35,nothing)

ax = Axis(fig[1,2],ylabel = L"\mathcal{P}", xlabel = L"\omega_c t",
 ylabelrotation=0.0, xlabelsize=25.0f0,ylabelsize=25.0f0, 
 title=L"\mathrm{TDD_2}", titlesize=20.0f0,
 ylabelvisible=false,yticklabelsvisible=false,
 xticks = ([0,1,2,3,4],[L"0",L"1",L"2",L"3",L"4"]))

lines!(ts , trρ , color=:blue )
band!(ts , trρ - Δρ , trρ + Δρ, color = (:blue, 0.2))

lines!(ts , trρ_dd2_c , color=:grey ,linestyle=:solid)
scatter!([ts[m] for m in ind2_c] , [trρ_dd2_c[m] for m in ind2_c], marker=:diamond,color=(:grey,0.2), strokecolor=:grey,strokewidth = 1.5, markersize=9)
band!(ts , trρ_dd2_c - Δρ_dd2_c , trρ_dd2_c + Δρ_dd2_c, color = (:grey, 0.2))

lines!(ts , trρ_dd2_b , color=(:grey,0.5)  ,linestyle=:solid)
scatter!([ts[m] for m in ind2_b] , [trρ_dd2_b[m] for m in ind2_b], marker=:rtriangle,color=(:grey,0.2), strokecolor=:grey,strokewidth = 1.5, markersize=9)
band!(ts , trρ_dd2_b - Δρ_dd2_b , trρ_dd2_b + Δρ_dd2_b, color = (:grey, 0.2))

lines!(ts , trρ_dd2_d , color=(:grey,0.2)  ,linestyle=:solid)
scatter!([ts[m] for m in ind2_d] , [trρ_dd2_d[m] for m in ind2_d], marker=:circ,color=(:grey,0.5), strokecolor=:grey,strokewidth = 1.5, markersize=9)
band!(ts , trρ_dd2_d - Δρ_dd2_d , trρ_dd2_d + Δρ_dd2_d, color = (:grey, 0.2))

ylims!(0.35,nothing)

Legend(fig[1,3], marks , [L"0.075", L"0.15" , L"0.3" ], L"\omega_c \tau", titlesize=25.0f0, labelsize=18.0f0)
display(fig)
CairoMakie.activate!()
CairoMakie.save(data_dir*"/fig.pdf", fig)


