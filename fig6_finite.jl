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

    return Convert_MultiFloat(H_tot/norm(H_tot, Inf))

end


function Propagate_Sequence_FP(seq::Vector, δ::Float64, Δ::Float64, H_disorder::Operator, H_dipole::Operator,N::Int, b::SpinBasis, FA::Float64, AM::Float64) # finite pulses 

    # seq[i][1] = axis 
    # seq[i][2] = angle 

    # δ = strength disorder 
    # Δ = strength dipole 
    # FA = potential flip angle errors 
    # AM = potential Axis-misspecification

    H_error = δ*H_disorder + Δ*H_dipole 

    U = exponential!(dense(-im*H_error*0.0).data)

    for n in 1:length(seq)
        U = exponential!(dense(Convert_MultiFloat(-im*seq[n][2]*(H_error + Rotation_Hamiltonian(N,seq[n][1], b, FA, AM)))).data)*U
    end
    return U 
end

function Convert_MultiFloat(H::Operator)
    basis = H.basis_l 
    return DenseOperator(basis,ComplexDF64.(dense(H).data))
end

data_dir = "/home/colin/Documents/JULIA/Codes persos/Platonic_sequences/Multisymmetrization/Codes 4 GitHub/Data_fig6/"

#################
# Generate Data #
################# 

# generators in axis-angle notation 

π2_x = ([1,0,0],pi/2)
π2_y = ([0,1,0],pi/2)
π2_z =  ([0,0,1],pi/2)

mπ2_x = (-[1,0,0],pi/2)
mπ2_y = (-[0,1,0],pi/2)
mπ2_z =  (-[0,0,1],pi/2)

π_x = ([1,0,0],pi)
π_y = ([0,1,0],pi)
π_z =  ([0,0,1],pi)

mπ_x = (-[1,0,0],pi)
mπ_y = (-[0,1,0],pi)
mπ_z =  (-[0,0,1],pi)

n1 = ([sqrt(2/3) , 0 , sqrt(1/3)] , pi)
n2p = (-[-sqrt(1/6) , sqrt(1/2) , sqrt(1/3)] ,pi)
n2m = (-[-sqrt(1/6) , -sqrt(1/2) , sqrt(1/3)] ,pi)

mn1 = (-[sqrt(2/3) , 0 , sqrt(1/3)] , pi)
mn2p = ([-sqrt(1/6) , sqrt(1/2) , sqrt(1/3)] ,pi)
mn2m = ([-sqrt(1/6) , -sqrt(1/2) , sqrt(1/3)] ,pi)

#sequences 

DROID60 = [π_x, π2_x ,  mπ2_y , mπ_x , mπ_x , π_x , π2_x , mπ2_y , mπ_x , mπ_x ,
 π_x , π2_x , mπ2_y , mπ_x , mπ_x , mπ_y , mπ2_y , π2_x , π_y , π_y , mπ_y , mπ2_y , 
 π2_x , π_y , π_y , mπ_y , mπ2_y , π2_x , π_y , π_y ,  mπ_y , π2_x , π2_y , π_y , mπ_y , 
  mπ_y , π2_x , π2_y , π_y , mπ_y ,  mπ_y , π2_x , π2_y , π_y , mπ_x, mπ_x, π2_y , π2_x , 
  π_x , mπ_x , mπ_x , π2_y , π2_x , π_x , mπ_x , mπ_x , π2_y , π2_x , π_x , mπ_y ]

yxx24 = [mπ2_y , π2_x , mπ2_x , π2_y , mπ2_x , mπ2_x , π2_y , mπ2_x , π2_x , mπ2_y , π2_x , π2_x , π2_y , mπ2_x , π2_x , mπ2_y , π2_x , π2_x , mπ2_y , π2_x , mπ2_x , π2_y , mπ2_x , mπ2_x  ]

TEDDY = [ n1 , n2p , n1 , n2p , n2p , n1 , n2p , n1 ]

TEDDY_refl = [ n1 , n2p , n1 , n2p , n2p , n1 , n2p , n1 , 
mn1 ,mn2p , mn1 , mn2p , mn2p , mn1 , mn2p , mn1 ]

# system 
jj = 1/2
bb = SpinBasis(jj)
J = [sigmax(bb)/2, -sigmay(bb)/2, -sigmaz(bb)/2] 
N = 3

logδ = LinRange(-5,0, 40)
logΔ = LinRange(-5,0,40)

δs = exp10.(logδ)
Δs = exp10.(logΔ)

data_DROID60 = Vector{Matrix}(undef, 3)
data_XYY24 = Vector{Matrix}(undef, 3)
data_TEDDY = Vector{Matrix}(undef, 3)
data_TEDDY_refl = Vector{Matrix}(undef, 3)

for n in 1:1
    data_DROID60[n] = Matrix{Float64}(undef, 40,40)
    data_XYY24[n] = Matrix{Float64}(undef, 40,40)
    data_TEDDY[n] = Matrix{Float64}(undef, 40,40)
    data_TEDDY_refl[n] = Matrix{Float64}(undef, 40,40)
end

data_0 = Matrix{Float64}(undef, 40,40)

Tmin = 8*pi

pulse_err = [0.0] # add pulse errors if desired 
nsample = 20

for p in 1:1
    for i in 1:40 
        Threads.@threads for j in 1:40

            moy_DROID60 = []
            moy_XYY24 = []

            moy0 = []
            moy_MD = []
            moy_MD2 = []

            for n in 1:nsample

                H_disorder = Disorder_Hamiltonian(N,bb)
                H_dipole = Dipole_Hamiltonian(N,bb)

                U_DROID60 = Propagate_Sequence_FP(DROID60 , δs[i] , Δs[j] , H_disorder, H_dipole , N , bb, 0.0, pulse_err[p])
                U_xyy24 = Propagate_Sequence_FP(yxx24, δs[i] , Δs[j] , H_disorder, H_dipole , N , bb, 0.0, pulse_err[p])
                U_0 = exp(-im*Tmin*(δs[i]*H_disorder + Δs[j]*H_dipole))

                U_MD = Propagate_Sequence_FP(TEDDY , δs[i] , Δs[j] , H_disorder, H_dipole , N , bb, 0.0, pulse_err[p])
                U_MD2 = Propagate_Sequence_FP(TEDDY_refl , δs[i] , Δs[j] , H_disorder, H_dipole , N , bb, 0.0, pulse_err[p])

                push!(moy_DROID60, log10(sqrt(abs(1 - (1/((2*jj+1)^N))*abs(tr(U_DROID60)))))) 
                push!(moy_XYY24, log10(sqrt(abs(1 - (1/((2*jj+1)^N))*abs(tr(U_xyy24))))))
                push!(moy0, log10(sqrt(abs(1 - (1/((2*jj+1)^N))*abs(tr(U_0))))))

                push!(moy_MD,log10(sqrt(abs(1 - (1/((2*jj+1)^N))*abs(tr(U_MD))))))
                push!(moy_MD2,log10(sqrt(abs(1 - (1/((2*jj+1)^N))*abs(tr(U_MD2))))))
                
            end
            
            data_0[i,j] = sum(moy0[n] for n in 1:nsample)/nsample

            data_DROID60[p][i,j] = sum(moy_DROID60[n] for n in 1:nsample)/nsample
            data_XYY24[p][i,j] = sum(moy_XYY24[n] for n in 1:nsample)/nsample

            data_TEDDY[p][i,j] = sum(moy_MD[n] for n in 1:nsample)/nsample
            data_TEDDY_refl[p][i,j] = sum(moy_MD2[n] for n in 1:nsample)/nsample
        end
        println("i = $i/40, p =$p/1")
    end

end

jldopen(data_dir * "spin$jj N=$N control finite.jld", "w") do file
    write(file, "data", data_0)
end
jldopen(data_dir * "spin$jj N=$N data_DROID60 finite.jld", "w") do file
    write(file, "data", data_DROID60)
end
jldopen(data_dir * "spin$jj N=$N yxx24 finite.jld", "w") do file
    write(file, "data", data_XYY24)
end

jldopen(data_dir * "spin$jj N=$N TEDDY finite.jld", "w") do file
    write(file, "data", data_TEDDY)
end
jldopen(data_dir * "spin$jj N=$N TEDDY_refl finite.jld", "w") do file
    write(file, "data", data_TEDDY_refl)
end

#############
# Load Data #
############# 

data_0 = load(data_dir * "spin$jj N=$N control finite.jld", "data")
data_DROID60 = load(data_dir * "spin$jj N=$N data_DROID60 finite.jld", "data")
data_XYY24 = load(data_dir * "spin$jj N=$N yxx24 finite.jld", "data")
data_TEDDY = load(data_dir * "spin$jj N=$N TEDDY finite.jld", "data")
data_TEDDY_refl = load(data_dir * "spin$jj N=$N TEDDY_refl finite.jld", "data")

logδ = LinRange(-5,0, 40)
logΔ = LinRange(-5,0,40)

#############
# Plot Data #
#############

colorsbright = [color4, color5 , color3 , color6, color1,color7]
colors = [color4darker, color5darker , color3darker , color6darker,color1darker,color7darker]
cr = Vector{Matrix}(undef, 6)
for n in 1:6
    cr[n] = Matrix{Any}(undef, 80,80)
    cr[n] .= colorsbright[n]
end
legends_marker = [[PolyElement(color=colorsbright[1], strokecolor=colors[1], strokewidth=2), LineElement(color=colors[1], linestyle=:dash)], 
[PolyElement(color=colorsbright[2], strokecolor=colors[2], strokewidth=2), LineElement(color=colors[2], linestyle=:dash)],
[PolyElement(color=colorsbright[5], strokecolor=colors[5], strokewidth=2), LineElement(color=colors[5], linestyle=:dash)],
[PolyElement(color=colorsbright[3], strokecolor=colors[3], strokewidth=2), LineElement(color=colors[3], linestyle=:dash)],
[PolyElement(color=colorsbright[4], strokecolor=colors[4], strokewidth=2), LineElement(color=colors[4], linestyle=:dash)]]


labsize = 30.0f0
ticklabelsize = 25.0f0

fig = Figure(;size=(600,600)) 

for p in 1:1
    ax = Axis3(fig[1,p], xlabel = L"\delta/\chi", ylabel = L"\Delta / \chi",zlabel = L"\overline{D}" ,
    xlabelsize = labsize, ylabelsize = labsize, zlabelsize = labsize , 
    xticks = ([-4,-2,0], [L"10^{-4}" , L"10^{-2}" , L"1"]), 
    yticks = ([-4,-2,0], [L"10^{-4}" , L"10^{-2}" , L"1"]), 
    zticks = ([-12,-9,-6,-3,0], [L"10^{-12}",L"10^{-9}",L"10^{-6}" , L"10^{-3}" , L"1"]), 
    xticklabelsize = ticklabelsize,yticklabelsize = ticklabelsize,zticklabelsize = ticklabelsize,
    protrusions = (90,0,0,0), 
    zlabelrotation = 0, zlabeloffset = 90.0, ylabeloffset=50.0)

    surface!(logδ[1:40], logΔ[1:40], [data_0[i,j] for i in 1:40, j in 1:40], color=cr[1], transparency = true, alpha = 0.9)
    surface!(logδ[1:40], logΔ[1:40], [data_TEDDY[p][i,j] for i in 1:40, j in 1:40], color=cr[2], transparency = false, alpha = 0.95)
    surface!(logδ[1:40], logΔ[1:40], [data_TEDDY_refl[p][i,j] for i in 1:40, j in 1:40], color=cr[5], transparency = false, alpha = 0.95)
    surface!(logδ[1:40], logΔ[1:40], [data_DROID60[p][i,j] for i in 1:40, j in 1:40], color=cr[3], transparency = true, alpha = 0.95)
    surface!(logδ[1:40], logΔ[1:40], [data_XYY24[p][i,j] for i in 1:40, j in 1:40], color=cr[4])

    ind1 = [1,8,16,24,32,40]
    ind2 = [1,8,16,24,32,40] # profile line

    for i in ind1
        lines!([logδ[i]], logΔ[1:40], data_0[i, 1:40], color=colors[1], linewidth=2.0, linestyle=:dash, transparency=false, alpha=0.9)
        lines!([logδ[i]], logΔ[1:40], data_TEDDY[p][i,1:40], color=colors[2], linewidth=2.0, transparency=false, alpha=0.95)
        lines!([logδ[i]], logΔ[1:40], data_TEDDY_refl[p][i,1:40], color=colors[5], linewidth=2.0, transparency=false, alpha=0.95)
        lines!([logδ[i]], logΔ[1:40], data_DROID60[p][i,1:40], color=colors[3], linewidth=2.0, transparency=true, alpha=0.95)
        lines!([logδ[i]], logΔ[1:40], data_XYY24[p][i, 1:40], color=colors[4], linewidth=2.0)

    end
    for i in ind2
        lines!(logδ[1:40], [logΔ[i]], data_0[1:40, i], color=colors[1], linewidth=2.0, linestyle=:dash, transparency=false, alpha=0.9)
        lines!(logδ[1:40], [logΔ[i]], data_TEDDY[p][1:40, i], color=colors[2], linewidth=2.0, transparency=false, alpha=0.95)
        lines!(logδ[1:40], [logΔ[i]], data_TEDDY_refl[p][1:40, i], color=colors[5], linewidth=2.0, transparency=false, alpha=0.95)
        lines!(logδ[1:40], [logΔ[i]], data_DROID60[p][1:40, i], color=colors[3], linewidth=2.0, transparency=true, alpha=0.95)
        lines!(logδ[1:40], [logΔ[i]], data_XYY24[p][1:40, i], color=colors[4], linewidth=2.0)

    end
end
Legend(fig[1,2], legends_marker[1:5], [L"\text{NoDD}", L"\text{TEDDY}" , L"\text{TEDDY}(\text{TEDDY})^{\dagger}" , L"\text{DROID60}" , L"\text{yxx24}"], framevisible = false , nbanks=1, tellheight=false)

display(fig)

save(data_dir*"fig_6 finite.png", fig, px_per_unit=10.0)
cd(data_dir)
fn = "fig_6 finite"
run(`mogrify -trim $fn.png`)


labsize = 40.0f0
ticklabelsize = 35.0f0
fig = Figure(;size=(600,600)) 

for p in 1:1
    ax = Axis3(fig[1,p], xlabel = L"\delta/\chi", ylabel = L"\Delta / \chi",zlabelvisible=false ,
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
    surface!(logδ[1:40], logΔ[1:40], [data_TEDDY[p][i,j] for i in 1:40, j in 1:40], color=cr[2], transparency = false, alpha = 0.95,shading=NoShading)
    surface!(logδ[1:40], logΔ[1:40], [data_TEDDY_refl[p][i,j] for i in 1:40, j in 1:40], color=cr[5], transparency = false, alpha = 0.95,shading=NoShading)
    surface!(logδ[1:40], logΔ[1:40], [data_DROID60[p][i,j] for i in 1:40, j in 1:40], color=cr[3], transparency = false, alpha = 0.95,shading=NoShading)
    surface!(logδ[1:40], logΔ[1:40], [data_XYY24[p][i,j] for i in 1:40, j in 1:40], color=cr[4],transparency=false,shading=NoShading)


    ind1 = [1,8,16,24,32,40]
    ind2 = [1,8,16,24,32,40] # profile line

    for i in ind1
        lines!([logδ[i]], logΔ[1:40], [-15 for i in 1:40], color=:black, linewidth=1.0, linestyle=:dash, transparency=false, alpha=0.9)
    end
    for i in ind2
        lines!(logδ[1:40], [logΔ[i]], [-15 for i in 1:40], color=:black, linewidth=1.0, linestyle=:dash, transparency=false, alpha=0.9)
    end
end
Legend(fig[2,1], legends_marker[1:5], [L"\text{NoDD}", L"\text{TEDDY}" , L"\text{TEDDY}(\text{TEDDY})^{\dagger}" , L"\text{DROID60}" , L"\text{yxx24}"], framevisible = false , nbanks=3, tellheight=true)

display(fig)

save(data_dir*"fig_6 finite (2).png", fig, px_per_unit=10.0) 
cd(data_dir)
fn = "fig_6 finite (2)"
run(`mogrify -trim $fn.png`)

