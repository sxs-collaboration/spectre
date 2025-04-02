# -*- perl -*-
#This file should be read by your evolution's DoMultipleRuns.input
#It defines the properties of the initial data that are relevant to do
#an evolution

#Compiled at mbot:/home/fs01/spec1168/spec
#Error checking disabled (to activate add -DDEBUG to CPPFLAGS)
#Code Revision ddabc19dbd98539df19d1c1b3a52ac43e1087325
#Linked on 2024-08-30T11:19:47-04:00
#Executed BnsIdSolver
#Centers of the neutron stars
@CenterNS1 = (-16.2,0,0);
@CenterNS2 = (16.1996,3.59764e-05,0);

$ID_d = 32.3996;

#Baryon masses of the neutron stars
$MassNS1 = 1.4958;
$MassNS2 = 1.4958;
$ADMmassNS1 = 1.35;
$ADMmassNS2 = 1.35;
#Radii of the stars
$r1 = 6.06997;
$r2 = 6.06997;

#Central baryon density of neutron stars
$CentralDensity1 = 0.00137887;
$CentralDensity2 = 0.00137887;
#Orbital parameters of the binary
$InitialOmega = 0.00801722; # deprecated

$ID_Omega0 = 0.00801722;

#Inital rate of shrinking
$ID_adot0 = -0.00008095;
#Equation of state
$EOS = 'SpectralGamma(GammaCoefs=2.000000,0.00000, 0.4029, -0.1008; \
GammaThermal=1.75; rho0=1.0118e-4; P0=3.3625e-7; RhoMax=0.005;)';
#Initial Adm Energy
$InitialAdmEnergy = 2.67601
