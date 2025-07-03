using QuantumOptics
using Makie 
using CairoMakie 
using GLMakie 
using Colors
using LinearAlgebra
using DoubleFloats
using ExponentialUtilities
using JLD

GLMakie.activate!()


color1 = RGB(255/255,153/255,153/255)
color1darker =  RGB(255/255,51/255,51/255)

color2 =  RGB(255/255,255/255,153/255)
color2darker =  RGB(255/255,255/255,51/255)

color3 = RGB(153/255,255/255,153/255)
color3darker = RGB(51/255,255/255,51/255)

color4 = RGB(153/255,204/255,255/255)
color4darker = RGB(51/255,153/255,255/255)

color5 = RGB(204/255,153/255,255/255)
color5darker = RGB(178/255,102/255,255/255)

color6 = RGB(255/255,204/255,153/255)
color6darker = RGB(255/255,153/255,51/255)

color7 = RGB(224/255,224/255,224/255)
color7darker = RGB(160/255,160/255,160/255)


function Rotation_Hamiltonian(N::Int, axis::Vector, b::SpinBasis, FA::Float64, AM::Float64)

    Id = one(b) # identity 

    J = [sigmax(b)/2, -sigmay(b)/2, -sigmaz(b)/2] # angular momentum vector

    rand_ax = [sqrt(2)/2, sqrt(2)/2,0]
    rand_ax = rand_ax/norm(rand_ax)

    axis_perp_1 = cross(rand_ax, axis)
    axis_perp_2 = cross(axis, axis_perp_1)


    H_Rot = (1 + FA)*(sqrt(1 - AM^2 - AM^2)*sum(axis[n]*J[n] for n in 1:3) + AM*sum(axis_perp_1[n]*J[n] for n in 1:3) + AM*sum(axis_perp_2[n]*J[n] for n in 1:3))

    H_tot = 0.0*Id 
    for n in 2:N 
        H_tot = tensor(Id,H_tot)
    end

    for n in 1:N 

        if n == 1
            O = H_Rot 
            for m in 2:N 
                O = tensor(O, Id)
            end
            H_tot += O 
        else 
            O = Id 
            for m in 2:n-1 
                O = tensor(O,Id)
            end
            O = tensor(O, H_Rot)
            for m in n+1:N 
                O = tensor(O, Id)
            end
            H_tot += O
        end
    end

    return H_tot
end

function Disorder_Hamiltonian(N::Int, b::SpinBasis)

    Id = one(b) # identity 

    J = [sigmax(b)/2, -sigmay(b)/2, -sigmaz(b)/2] # angular momentum vector

    H_tot = 0.0*Id 
    for n in 2:N 
        H_tot = tensor(Id,H_tot)
    end

    for n in 1:N 

        if n == 1
            O = rand()*J[3] 
            for m in 2:N 
                O = tensor(O, Id)
            end
            H_tot += O 
        else 
            O = rand()*Id 
            for m in 2:n-1 
                O = tensor(O,Id)
            end
            O = tensor(O, J[3])
            for m in n+1:N 
                O = tensor(O, Id)
            end
            H_tot += O
        end
    end

    return Convert_MultiFloat(H_tot/norm(H_tot, Inf))
end


function Dipole_Hamiltonian(N::Int, b::SpinBasis)

    Id = one(b) # identity 

    J = [sigmax(b)/2, -sigmay(b)/2, -sigmaz(b)/2] # angular momentum vector

    H_tot = 0.0*Id
    for n in 2:N 
        H_tot = tensor(Id,H_tot)
    end


    for n in 1:N-1 
        for m in n+1:N

            A = rand()

            if n == 1

                O = A*J[3]
                for k in n+1:m-1 
                    O = tensor(O,Id)
                end
                O = tensor(O, J[3])
                for k in m+1:N 
                    O = tensor(O, Id)
                end

                H_tot += 2*O 

            else

                O = A*Id 
                for k in 2:n-1 
                    O = tensor(O,Id)
                end
                O = tensor(O, J[3])
                for k in n+1:m-1 
                    O = tensor(O, Id)
                end
                O = tensor(O, J[3])
                for k in m+1:N 
                    O = tensor(O, Id)
                end

                H_tot += 2*O 

            end

            if n == 1

                O = A*J[1]
                for k in n+1:m-1 
                    O = tensor(O,Id)
                end
                O = tensor(O, J[1])
                for k in m+1:N 
                    O = tensor(O, Id)
                end

                H_tot -= O 

            else

                O = A*Id 
                for k in 2:n-1 
                    O = tensor(O,Id)
                end
                O = tensor(O, J[1])
                for k in n+1:m-1 
                    O = tensor(O, Id)
                end
                O = tensor(O, J[1])
                for k in m+1:N 
                    O = tensor(O, Id)
                end

                H_tot -= O 

            end

            if n == 1

                O = A*J[2]
                for k in n+1:m-1 
                    O = tensor(O,Id)
                end
                O = tensor(O, J[2])
                for k in m+1:N 
                    O = tensor(O, Id)
                end

                H_tot -= O 

            else

                O = A*Id 
                for k in 2:n-1 
                    O = tensor(O,Id)
                end
                O = tensor(O, J[2])
                for k in n+1:m-1 
                    O = tensor(O, Id)
                end
                O = tensor(O, J[2])
                for k in m+1:N 
                    O = tensor(O, Id)
                end

                H_tot -= O 

            end

        end
    end
    return Convert_MultiFloat(H_tot/norm(H_tot,Inf))
end


function Propagate_Sequence_IP(τ, seq::Vector, δ, Δ, H_disorder::Operator, H_dipole::Operator) # propagate sequence (all sequences but not DROID-60)

    # seq[i] = propagator pulse 

    H_error = Convert_MultiFloat(δ*(H_disorder) + Δ*(H_dipole))

    U = exponential!(dense(-im*0*H_error).data)
    U0 = exponential!(dense(-im*τ*(H_error)).data)

    for n in 1:length(seq)
        U = seq[n]*U0*U
    end

    return U 
end

function Propagate_Sequence_IPDroid(τ, seq::Vector, δ, Δ, H_disorder::Operator, H_dipole::Operator) # propagate DROID-60 (because the π/2 pulses should be applied with no waiting time in-between)

    # seq[i] = propagator pulse 

    H_error = δ*H_disorder + Δ*H_dipole 

    U = exponential!(dense(-im*0.0*(H_error)).data)
    U0 = exponential!(dense(-im*τ*(H_error)).data)

    for n in 1:length(seq)
        U = U0*U
        for m in 1:length(seq[n])
            U = seq[n][m]*U
        end
    end

    return U 
end

function Convert_MultiFloat(H::Operator)
    basis = H.basis_l 
    return DenseOperator(basis,ComplexDF64.(dense(H).data))
end

data_dir = "/home/colin/Documents/JULIA/Codes persos/Platonic_sequences/Multisymmetrization/Codes 4 GitHub/Data_fig4/"

#################
# Generate Data #
################# 

jj = 1/2
b = SpinBasis(jj)
N = 3

# generators 

π2_x = exponential!(dense(-im*pi/2*Convert_MultiFloat(Rotation_Hamiltonian(N,[1,0,0],b,0.0,0.0))).data)
π2_y = exponential!(dense(-im*pi/2*Convert_MultiFloat(Rotation_Hamiltonian(N,[0,1,0],b,0.0,0.0))).data)
π2_z =  exponential!(dense(-im*pi/2*Convert_MultiFloat(Rotation_Hamiltonian(N,[0,0,1],b,0.0,0.0))).data)

mπ2_x = exponential!(dense(-im*pi/2*Convert_MultiFloat(Rotation_Hamiltonian(N,-[1,0,0],b,0.0,0.0))).data)
mπ2_y = exponential!(dense(-im*pi/2*Convert_MultiFloat(Rotation_Hamiltonian(N,-[0,1,0],b,0.0,0.0))).data)
mπ2_z =  exponential!(dense(-im*pi/2*Convert_MultiFloat(Rotation_Hamiltonian(N,-[0,0,1],b,0.0,0.0))).data)

π_x = exponential!(dense(-im*pi*Convert_MultiFloat(Rotation_Hamiltonian(N,[1,0,0],b,0.0,0.0))).data)
π_y = exponential!(dense(-im*pi*Convert_MultiFloat(Rotation_Hamiltonian(N,[0,1,0],b,0.0,0.0))).data)
π_z =  exponential!(dense(-im*pi*Convert_MultiFloat(Rotation_Hamiltonian(N,[0,0,1],b,0.0,0.0))).data)

mπ_x = exponential!(dense(-im*pi*Convert_MultiFloat(Rotation_Hamiltonian(N,-[1,0,0],b,0.0,0.0))).data)
mπ_y = exponential!(dense(-im*pi*Convert_MultiFloat(Rotation_Hamiltonian(N,-[0,1,0],b,0.0,0.0))).data)
mπ_z =  exponential!(dense(-im*pi*Convert_MultiFloat(Rotation_Hamiltonian(N,-[0,0,1],b,0.0,0.0))).data)

n1 = exponential!(dense(Convert_MultiFloat(-im*pi*Rotation_Hamiltonian(N,[sqrt(2/3) , 0 , sqrt(1/3)],b,0.0,0.0))).data)
n2p = exponential!(dense(Convert_MultiFloat(-im*pi*Rotation_Hamiltonian(N,-[-sqrt(1/6) , sqrt(1/2) , sqrt(1/3)],b,0.0,0.0))).data)
n2m = exponential!(dense(Convert_MultiFloat(-im*pi*Rotation_Hamiltonian(N,-[-sqrt(1/6) , -sqrt(1/2) , sqrt(1/3)],b,0.0,0.0))).data)

mn1 = exponential!(dense(Convert_MultiFloat(-im*pi*Rotation_Hamiltonian(N,-[sqrt(2/3) , 0 , sqrt(1/3)],b,0.0,0.0))).data)
mn2p = exponential!(dense(Convert_MultiFloat(-im*pi*Rotation_Hamiltonian(N,[-sqrt(1/6) , sqrt(1/2) , sqrt(1/3)],b,0.0,0.0))).data)
mn2m = exponential!(dense(Convert_MultiFloat(-im*pi*Rotation_Hamiltonian(N,[-sqrt(1/6) , -sqrt(1/2) , sqrt(1/3)],b,0.0,0.0))).data)

DROID60 = [[π_x], [π2_x ,  mπ2_y] , [mπ_x] , [mπ_x] , [π_x] , [π2_x , mπ2_y] , [mπ_x] , [mπ_x] ,
 [π_x] , [π2_x , mπ2_y] , [mπ_x] , [mπ_x] , [mπ_y] , [mπ2_y , π2_x] , [π_y] , [π_y] , [mπ_y] , [mπ2_y , 
 π2_x] , [π_y] , [π_y] , [mπ_y] , [mπ2_y , π2_x] , [π_y] , [π_y] ,  [mπ_y] , [π2_x , π2_y] , [π_y] , [mπ_y] , 
  [mπ_y] , [π2_x , π2_y] , [π_y] , [mπ_y] ,  [mπ_y] , [π2_x , π2_y] , [π_y] , [mπ_x], [mπ_x], [π2_y , π2_x] , 
  [π_x] , [mπ_x] , [mπ_x] , [π2_y , π2_x] , [π_x] , [mπ_x] , [mπ_x] , [π2_y , π2_x] , [π_x] , [mπ_y] ]

yxx24 = [mπ2_y , π2_x , mπ2_x , π2_y , mπ2_x , mπ2_x , π2_y , mπ2_x , π2_x , mπ2_y , π2_x , π2_x , π2_y , mπ2_x , π2_x , mπ2_y , π2_x , π2_x , mπ2_y , π2_x , mπ2_x , π2_y , mπ2_x , mπ2_x  ]

TEDDY = [ n1 , n2p , n1 , n2p , n2p , n1 , n2p , n1 ]

logδ = LinRange(-5,0, 40)
logΔ = LinRange(-5,0,40)

δs = exp10.(logδ)
Δs = exp10.(logΔ)

data_DROID60 = Matrix{Float64}(undef, 40,40)
data_XYY24 = Matrix{Float64}(undef, 40,40)
data_TEDDY = Matrix{Float64}(undef, 40,40)
data_0 = Matrix{Float64}(undef, 40,40)

τ=1.0

Tmin = 8*τ

nsample = 20

for i in 1:40 
    Threads.@threads for j in 1:40

        moy_DROID60 = []
        moy_XYY24 = []
        moy0 = []
        moy_MD = []

        for n in 1:nsample

            H_disorder = Disorder_Hamiltonian(N,b)
            H_dipole = Dipole_Hamiltonian(N,b)

            U_DROID60 = Propagate_Sequence_IPDroid(τ ,DROID60 , δs[i] , Δs[j] , H_disorder, H_dipole)
            U_xyy24 = Propagate_Sequence_IP(τ ,yxx24, δs[i] , Δs[j] , H_disorder, H_dipole)
            U_00 = exp(-im*Tmin*(δs[i]*H_disorder + Δs[j]*H_dipole))

            U_MD = Propagate_Sequence_IP(τ ,TEDDY, δs[i] , Δs[j] , H_disorder, H_dipole)

            push!(moy_DROID60, log10(sqrt(abs(1 - (1/((2*jj+1)^N))*abs(tr(U_DROID60)))))) 
            push!(moy_XYY24, log10(sqrt(abs(1 - (1/((2*jj+1)^N))*abs(tr(U_xyy24))))))
            push!(moy0, log10(sqrt(abs(1 - (1/((2*jj+1)^N))*abs(tr(U_00))))))
            push!(moy_MD,log10(sqrt(abs(1 - (1/((2*jj+1)^N))*abs(tr(U_MD))))))
            
        end

        
        data_0[i,j] = sum(moy0[n] for n in 1:nsample)/nsample

        data_DROID60[i,j] = sum(moy_DROID60[n] for n in 1:nsample)/nsample
        data_XYY24[i,j] = sum(moy_XYY24[n] for n in 1:nsample)/nsample
        data_TEDDY[i,j] = sum(moy_MD[n] for n in 1:nsample)/nsample
    end
    println("i = $i/40")
end

jldopen(data_dir * "spin$jj N=$N control ideal.jld", "w") do file
    write(file, "data", data_0)
end
jldopen(data_dir * "spin$jj N=$N data_DROID60 ideal.jld", "w") do file
    write(file, "data", data_DROID60)
end
jldopen(data_dir * "spin$jj N=$N yxx24 ideal.jld", "w") do file
    write(file, "data", data_XYY24)
end

jldopen(data_dir * "spin$jj N=$N TEDDY ideal.jld", "w") do file
    write(file, "data", data_TEDDY)
end

#############
# Load Data #
############# 

data_0 = load(data_dir * "spin$jj N=$N control ideal.jld", "data")
data_DROID60 = load(data_dir * "spin$jj N=$N data_DROID60 ideal.jld", "data")
data_XYY24 = load(data_dir * "spin$jj N=$N yxx24 ideal.jld", "data")
data_TEDDY = load(data_dir * "spin$jj N=$N TEDDY ideal.jld", "data")

logδ = LinRange(-5,0, 40)
logΔ = LinRange(-5,0,40)

#############
# Plot Data #
#############

colorsbright = [color4, color5 , color3 , color6, color1]
colors = [color4darker, color5darker , color3darker , color6darker,color1darker]

cr = Vector{Matrix}(undef, 5)
for n in 1:5
    cr[n] = Matrix{Any}(undef, 80,80)
    cr[n] .= colorsbright[n]
end
legends_marker = []
for i in 1:5
    push!(legends_marker, [PolyElement(color=colorsbright[i], strokecolor=colors[i], strokewidth=2), LineElement(color=colors[i], linestyle=:dash)])
end

labsize = 30.0f0
ticklabelsize = 25.0f0
fig = Figure(;size=(600,600))

ax = Axis3(fig[1,1], xlabel = L"\tau\delta", ylabel = L"\tau \Delta",zlabel = L"\overline{D}" ,
xlabelsize = labsize, ylabelsize = labsize, zlabelsize = labsize ,
xticks = ([-4,-2,0], [L"10^{-4}" , L"10^{-2}" , L"1"]),
yticks = ([-4,-2,0], [L"10^{-4}" , L"10^{-2}" , L"1"]),
zticks = ([-12,-9,-6,-3,0], [L"10^{-12}",L"10^{-9}",L"10^{-6}" , L"10^{-3}" , L"1"]),
xticklabelsize = ticklabelsize,yticklabelsize = ticklabelsize,zticklabelsize = ticklabelsize,
protrusions = (90,0,0,0),
zlabelrotation = 0, zlabeloffset = 90.0, ylabeloffset=50.0)

surface!(logδ[1:40], logΔ[1:40], [data_0[i,j] for i in 1:40, j in 1:40], color=cr[1], transparency = true, alpha = 0.9)
surface!(logδ[1:40], logΔ[1:40], [data_TEDDY[i,j] for i in 1:40, j in 1:40], color=cr[2], transparency = false, alpha = 0.95)
surface!(logδ[1:40], logΔ[1:40], [data_DROID60[i,j] for i in 1:40, j in 1:40], color=cr[3], transparency = true, alpha = 0.95)
surface!(logδ[1:40], logΔ[1:40], [data_XYY24[i,j] for i in 1:40, j in 1:40], color=cr[4])

ind1 = [1,8,16,24,32,40]
ind2 = [1,8,16,24,32,40] # profile line

for i in ind1
    lines!([logδ[i]], logΔ[1:40], data_0[i, 1:40], color=colors[1], linewidth=2.0, linestyle=:dash, transparency=false, alpha=0.9)
    lines!([logδ[i]], logΔ[1:40], data_TEDDY[i,1:40], color=colors[2], linewidth=2.0, transparency=false, alpha=0.95)
    lines!([logδ[i]], logΔ[1:40], data_DROID60[i,1:40], color=colors[3], linewidth=2.0, transparency=true, alpha=0.95)
    lines!([logδ[i]], logΔ[1:40], data_XYY24[i, 1:40], color=colors[4], linewidth=2.0)
end
for i in ind2
    lines!(logδ[1:40], [logΔ[i]], data_0[1:40, i], color=colors[1], linewidth=2.0, linestyle=:dash, transparency=false, alpha=0.9)
    lines!(logδ[1:40], [logΔ[i]], data_TEDDY[1:40, i], color=colors[2], linewidth=2.0, transparency=false, alpha=0.95)
    lines!(logδ[1:40], [logΔ[i]], data_DROID60[1:40, i], color=colors[3], linewidth=2.0, transparency=true, alpha=0.95)
    lines!(logδ[1:40], [logΔ[i]], data_XYY24[1:40, i], color=colors[4], linewidth=2.0)
end

Legend(fig[1,2], legends_marker[1:4], [L"\text{NoDD}", L"\text{TEDDY}" , L"\text{DROID60}" , L"\text{yxx24}"], framevisible = false , nbanks=1, tellheight=false)

display(fig)

save(data_dir*"fig_4 ideal.png", fig, px_per_unit=10.0)
cd(data_dir)
fn = "fig_4 ideal"
run(`mogrify -trim $fn.png`)


labsize = 40.0f0
ticklabelsize = 35.0f0
fig = Figure(;size=(600,600))

ax = Axis3(fig[1,1], xlabel = L"\tau\delta", ylabel = L"\tau \Delta",zlabelvisible=false ,
zticklabelsvisible=false, zticksvisible=false, zspinesvisible=false,
xlabelsize = labsize, ylabelsize = labsize, zlabelsize = labsize ,
xticks = ([-4,-2,0], [L"10^{-4}" , L"10^{-2}" , L"1"]),
yticks = ([-4,-2,0], [L"10^{-4}" , L"10^{-2}" , L"1"]),
zticks = ([-12,-9,-6,-3,0], [L"10^{-12}",L"10^{-9}",L"10^{-6}" , L"10^{-3}" , L"1"]),
xticklabelsize = ticklabelsize,yticklabelsize = ticklabelsize,zticklabelsize = ticklabelsize,
protrusions = (90,0,0,0),
zlabelrotation = 0, zlabeloffset = 90.0,perspectiveness = 0.0,elevation=-pi/2,azimuth=0*pi,
xlabelrotation=0, xlabeloffset=80)

surface!(logδ[1:40], logΔ[1:40], [data_0[i,j] for i in 1:40, j in 1:40], color=cr[1], transparency = false, alpha = 0.9,shading=NoShading)
surface!(logδ[1:40], logΔ[1:40], [data_TEDDY[i,j] for i in 1:40, j in 1:40], color=cr[2], transparency = false, alpha = 0.95,shading=NoShading)
surface!(logδ[1:40], logΔ[1:40], [data_DROID60[i,j] for i in 1:40, j in 1:40], color=cr[3], transparency = false, alpha = 0.95,shading=NoShading)
surface!(logδ[1:40], logΔ[1:40], [data_XYY24[i,j] for i in 1:40, j in 1:40], color=cr[4],transparency=false,shading=NoShading)

ind1 = [1,8,16,24,32,40]
ind2 = [1,8,16,24,32,40] # profile line

for i in ind1
    lines!([logδ[i]], logΔ[1:40], [-15 for i in 1:40], color=:black, linewidth=1.0, linestyle=:dash, transparency=false, alpha=0.9)
end
for i in ind2
    lines!(logδ[1:40], [logΔ[i]], [-15 for i in 1:40], color=:black, linewidth=1.0, linestyle=:dash, transparency=false, alpha=0.9)
end

Legend(fig[2,1], legends_marker[1:4], [L"\text{NoDD}", L"\text{TEDDY}" , L"\text{DROID60}" , L"\text{yxx24}"], framevisible = false , nbanks=4, tellheight=true)

display(fig)

save(data_dir*"fig_4 ideal (2).png", fig, px_per_unit=10.0)
cd(data_dir)
fn = "fig_4 ideal (2)"
run(`mogrify -trim $fn.png`)
