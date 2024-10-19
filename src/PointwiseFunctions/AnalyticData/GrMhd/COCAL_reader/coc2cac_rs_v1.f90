!_____________________________________________________________________________
!
!    CACTUS READER OF COCAL ROTATING STAR IN CF FORMALISM 
!_____________________________________________________________________________
!
include './phys_constant.f90'
include './def_matter_parameter.f90'
include './def_quantities.f90'
include './def_bh_parameter.f90'
include './def_peos_parameter.f90'
include './make_array_2d.f90'
include './make_array_3d.f90'
include './grid_parameter.f90'
include './interface_modules_cartesian.f90' 
include './coordinate_grav_r.f90' 
include './coordinate_grav_phi.f90' 
include './coordinate_grav_theta.f90' 
include './coordinate_grav_extended.f90' 
include './trigonometry_grav_theta.f90' 
include './trigonometry_grav_phi.f90' 
include './interface_IO_input_CF_grav_export.f90' 
include './interface_IO_input_WL_grav_export_hij.f90' 
include './interface_IO_input_grav_export_Kij.f90' 
include './interface_IO_input_CF_star_export.f90' 
include './interface_invhij_WL_export.f90' 
include './interface_index_vec_down2up_export.f90' 
include './interface_interpo_gr2fl_metric_CF_export.f90' 
include './interface_IO_input_grav_export_Ai.f90' 
include './interface_IO_input_grav_export_Faraday.f90' 
include './interface_IO_input_star4ve_export.f90' 
include './interface_interpo_gr2fl_export.f90' 
include './interface_interpo_lag4th_2Dsurf.f90' 
include './IO_input_CF_grav_export.f90'
include './IO_input_WL_grav_export_hij.f90'
include './IO_input_grav_export_Kij.f90'
include './IO_input_CF_star_export.f90'
include './invhij_WL_export.f90'
include './index_vec_down2up_export.f90'
include './interpo_gr2fl_metric_CF_export.f90'
include './IO_input_grav_export_Ai.f90'
include './IO_input_grav_export_Faraday.f90'
include './IO_input_star4ve_export.f90'
include './interpo_gr2fl_export.f90'
include './interpo_gr2cgr_4th.f90'
include './interpo_fl2cgr_4th_export.f90'
include './interpo_lag4th_2Dsurf.f90' 
include './lagint_4th.f90' 
include './peos_initialize.f90' 
include './peos_q2hprho.f90' 
include './peos_lookup.f90' 
!
!_____________________________________________________________________________
PROGRAM coc2cac
  use phys_constant
  use grid_parameter
  use interface_modules_cartesian
  use coordinate_grav_r
  use coordinate_grav_phi
  use coordinate_grav_theta
  use coordinate_grav_extended
  use trigonometry_grav_theta
  use trigonometry_grav_phi
  use interface_IO_input_CF_grav_export
  use interface_IO_input_CF_star_export
  use interface_IO_input_grav_export_Kij
!  use interface_excurve_CF_gridpoint_export
  use interface_interpo_gr2fl_metric_CF_export
  implicit none
  integer :: iAB 
  character(30) :: char1
  character*400 :: dir_path
  real(8) :: xcac, ycac, zcac
  real(8) :: xcoc, ycoc, zcoc
  real(8) :: emdca, omefca, psica, alphca, bvxdca, bvydca, bvzdca, psi4ca, psif4ca
  real(8) :: hca, preca, rhoca, eneca, epsca
  real(8) :: kxxca, kxyca, kxzca, kyyca, kyzca, kzzca
  real(8) :: vxu, vyu, vzu
  real(8) :: bxcor, bycor, bzcor, bvxdfca, bvydfca, bvzdfca, psifca, alphfca
  real(8) :: gxx, gxy, gxz, gyy, gyz, gzz, kxx1, kxy1, kxz1, kyy1, kyz1, kzz1
  real(8) :: ome, ber, radi
!
  real(8), pointer :: emd(:,:,:), omef(:,:,:), rs(:,:)
  real(8), pointer :: psif(:,:,:), alphf(:,:,:), bvxdf(:,:,:), bvydf(:,:,:), bvzdf(:,:,:)
  real(8), pointer :: psi(:,:,:) , alph(:,:,:) , bvxd(:,:,:) , bvyd(:,:,:) , bvzd(:,:,:)
  real(8), pointer :: kxx(:,:,:), kxy(:,:,:) , kxz(:,:,:) , kyy(:,:,:) , kyz(:,:,:), kzz(:,:,:)
!
  gxx=0.0d0; gxy=0.0d0; gxz=0.0d0; gyy=0.0d0; gyz=0.0d0; gzz=0.0d0
  kxx1=0.0d0; kxy1=0.0d0; kxz1=0.0d0; kyy1=0.0d0; kyz1=0.0d0; kzz1=0.0d0
  kxxca=0.0d0; kxyca=0.0d0; kxzca=0.0d0; kyyca=0.0d0; kyzca=0.0d0; kzzca=0.0d0

  !TODO remove this
  !dir_path="/home/astro/mundim/tmp/ET_2014_05_wheeler/Cactus/repos/Cocal/standalone/Cocal/ID_BNS"
  !dir_path="../../standalone/Cocal/ID_BNS"
  !dir_path='.'
  write(*,*) "Reading initial data from..."
  open(70,file='PATH2ID.txt',status='old')
  read(70,'(i1)') iAB
  read(70,'(a400)') dir_path
  close(70)
  write(*,*) dir_path 

! -- Read parameters
  call read_parameter_cactus(dir_path)
  call peos_initialize_cactus(dir_path)
  call grid_r
  call grid_theta
  call trig_grav_theta
  call grid_phi
  call allocate_trig_grav_mphi
  call trig_grav_phi
  call grid_extended
!
!    write(6,'(6i5)') nrg, ntg, npg, nrf, ntf, npf
!  rr3 = 0.7d0*(rgout - rgmid)
!  dis_cm = dis

  allocate (  emd(0:nrf,0:ntf,0:npf))
  allocate ( omef(0:nrf,0:ntf,0:npf))
  allocate ( psif(0:nrf,0:ntf,0:npf))
  allocate (alphf(0:nrf,0:ntf,0:npf))
  allocate (bvxdf(0:nrf,0:ntf,0:npf))
  allocate (bvydf(0:nrf,0:ntf,0:npf))
  allocate (bvzdf(0:nrf,0:ntf,0:npf))
  allocate (   rs(0:ntf,0:npf))
  allocate (  psi(0:nrg,0:ntg,0:npg))
  allocate ( alph(0:nrg,0:ntg,0:npg))
  allocate ( bvxd(0:nrg,0:ntg,0:npg))
  allocate ( bvyd(0:nrg,0:ntg,0:npg))
  allocate ( bvzd(0:nrg,0:ntg,0:npg))
  allocate (  kxx(0:nrg,0:ntg,0:npg))
  allocate (  kxy(0:nrg,0:ntg,0:npg))
  allocate (  kxz(0:nrg,0:ntg,0:npg))                                                                                                              
  allocate (  kyy(0:nrg,0:ntg,0:npg))
  allocate (  kyz(0:nrg,0:ntg,0:npg))
  allocate (  kzz(0:nrg,0:ntg,0:npg))
!  allocate (  axx(0:nrg,0:ntg,0:npg))
!  allocate (  axy(0:nrg,0:ntg,0:npg))
!  allocate (  axz(0:nrg,0:ntg,0:npg))
!  allocate (  ayy(0:nrg,0:ntg,0:npg))
!  allocate (  ayz(0:nrg,0:ntg,0:npg))
!  allocate (  azz(0:nrg,0:ntg,0:npg))

  emd=0.0d0;  rs  =0.0d0;  omef=0.0d0
  psi=0.0d0;  alph=0.0d0;  bvxd=0.0d0;  bvyd=0.0d0;  bvzd=0.0d0
  kxx=0.0d0;  kxy =0.0d0;  kxz =0.0d0;   kyy=0.0d0;   kyz=0.0d0;   kzz=0.0d0

  call IO_input_CF_grav_export(trim(dir_path)//"/rnsgra_3D.las",psi,alph,bvxd,bvyd,bvzd)

  call IO_input_CF_star_export(trim(dir_path)//"/rnsflu_3D.las",emd,rs,omef,ome,ber,radi)

!  call excurve_CF_gridpoint_export(alph,bvxd,bvyd,bvzd, & 
!     &    axx, axy, axz, ayy, ayz, azz)

  call IO_input_grav_export_Kij(trim(dir_path)//"/rnsgra_Kij_3D.las",kxx,kxy,kxz,kyy,kyz,kzz)

  call interpo_gr2fl_metric_CF_export(alph, psi, bvxd, bvyd, bvzd, &
        &    alphf, psif, bvxdf, bvydf, bvzdf, rs)


  write(6,'(2e20.12)') emd(0,0,0), omef(0,0,0)
  write(6,'(3e20.12)') ome, ber, radi
!
  write(6,'(a56)', ADVANCE = "NO") "Give cartesian coordinates (x,y,z) separated by a space:"
  read(5,*) xcac,ycac,zcac
  write(6,'(a23,3e20.12)') "Point given wrt CACTUS:", xcac,ycac,zcac
  write(6,'(a20,1e20.12)') "Cocal radius scale :", radi
  xcoc = xcac/(radi)
  ycoc = ycac/(radi)
  zcoc = zcac/(radi)
  write(6,'(a23,3e20.12)') "Point given wrt COCAL:", xcoc,ycoc,zcoc


  call interpo_gr2cgr_4th(psi , psica , xcoc, ycoc, zcoc)
  call interpo_gr2cgr_4th(alph, alphca, xcoc, ycoc, zcoc)
  call interpo_gr2cgr_4th(bvxd, bvxdca, xcoc, ycoc, zcoc)
  call interpo_gr2cgr_4th(bvyd, bvydca, xcoc, ycoc, zcoc)
  call interpo_gr2cgr_4th(bvzd, bvzdca, xcoc, ycoc, zcoc)

  call interpo_gr2cgr_4th(kxx , kxxca , xcoc, ycoc, zcoc)
  call interpo_gr2cgr_4th(kxy , kxyca , xcoc, ycoc, zcoc)
  call interpo_gr2cgr_4th(kxz , kxzca , xcoc, ycoc, zcoc)
  call interpo_gr2cgr_4th(kyy , kyyca , xcoc, ycoc, zcoc)
  call interpo_gr2cgr_4th(kyz , kyzca , xcoc, ycoc, zcoc)
  call interpo_gr2cgr_4th(kzz , kzzca , xcoc, ycoc, zcoc)

  call interpo_fl2cgr_4th_export(emd  , emdca   , xcoc, ycoc, zcoc, rs)
  call interpo_fl2cgr_4th_export(omef , omefca  , xcoc, ycoc, zcoc, rs)
  call interpo_fl2cgr_4th_export(psif , psifca  , xcoc, ycoc, zcoc, rs)
  call interpo_fl2cgr_4th_export(alphf, alphfca , xcoc, ycoc, zcoc, rs)
  call interpo_fl2cgr_4th_export(bvxdf, bvxdfca , xcoc, ycoc, zcoc, rs)
  call interpo_fl2cgr_4th_export(bvydf, bvydfca , xcoc, ycoc, zcoc, rs)
  call interpo_fl2cgr_4th_export(bvzdf, bvzdfca , xcoc, ycoc, zcoc, rs)

  bxcor = bvxdfca + omefca*(-ycoc)
  bycor = bvydfca + omefca*(xcoc)
  bzcor = bvzdfca
  psi4ca = psica**4
  psif4ca = psifca**4

  if (dabs(emdca) > 1.0d-14) then
    vxu = bxcor/alphfca 
    vyu = bycor/alphfca
    vzu = bzcor/alphfca
  else
    emdca=0.0d0
    vxu=0.0d0; vyu=0.0d0; vzu=0.0d0
  end if

  gxx = psi4ca
  gxy = 0.0d0
  gxz = 0.0d0
  gyy = psi4ca
  gyz = 0.0d0
  gzz = psi4ca

  kxx1 = psi4ca*kxxca/(radi)
  kxy1 = psi4ca*kxyca/(radi)
  kxz1 = psi4ca*kxzca/(radi)
  kyy1 = psi4ca*kyyca/(radi)
  kyz1 = psi4ca*kyzca/(radi)
  kzz1 = psi4ca*kzzca/(radi)

  call peos_q2hprho(emdca, hca, preca, rhoca, eneca)

  epsca = eneca/rhoca - 1.0d0

  write(6,'(a6,e20.12)') "psi  =", psica
  write(6,'(a6,e20.12)') "alph =", alphca
  write(6,'(a6,e20.12)') "bvxd =", bvxdca
  write(6,'(a6,e20.12)') "bvyd =", bvydca
  write(6,'(a6,e20.12)') "bvzd =", bvzdca
  write(6,'(a6,e20.12)') "Radi =", radi
  write(6,'(a6,e20.12)') "Omeg =", ome/radi
  write(6,'(a6,e20.12)') "emd  =", emdca
  write(6,'(a6,e20.12)') "h    =", hca
  write(6,'(a6,e20.12)') "pre  =", preca
  write(6,'(a6,e20.12)') "rho  =", rhoca
  write(6,'(a6,e20.12)') "ene  =", eneca
  write(6,'(a6,e20.12)') "eps  =", epsca
!
  write(6,'(a18)') "Kij at gridpoints:"
  write(6,'(3e20.12)') kxx1, kxy1, kxz1
  write(6,'(3e20.12)') kxy1, kyy1, kyz1
  write(6,'(3e20.12)') kxz1, kyz1, kzz1

  write(6,'(a13)') "v^i eulerian:"
  write(6,'(a6,e20.12)') "vxu  =", vxu
  write(6,'(a6,e20.12)') "vyu  =", vyu
  write(6,'(a6,e20.12)') "vzu  =", vzu

  write(6,'(a16)') "Deallocating...."
  deallocate(  emd)   
  deallocate( omef)   
  deallocate( psif)    
  deallocate(alphf)    
  deallocate(bvxdf)    
  deallocate(bvydf)    
  deallocate(bvzdf)    
  deallocate(   rs)    
  deallocate(  psi)  
  deallocate( alph) 
  deallocate( bvxd)  
  deallocate( bvyd)  
  deallocate( bvzd)  
  deallocate(  kxx)  
  deallocate(  kxy)  
  deallocate(  kxz)  
  deallocate(  kyy)  
  deallocate(  kyz)  
  deallocate(  kzz)  
!
END PROGRAM coc2cac
