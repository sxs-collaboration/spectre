!_____________________________________________________________________________
!
!    CACTUS READER OF COCAL MAGNETIZED ROTATING STAR IN WAVELESS FORMALISM 
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
  use interface_IO_input_WL_grav_export_hij
  use interface_IO_input_grav_export_Kij
  use interface_IO_input_CF_star_export
  use interface_invhij_WL_export
  use interface_index_vec_down2up_export
  use interface_interpo_gr2fl_metric_CF_export
  use interface_IO_input_grav_export_Ai
  use interface_IO_input_grav_export_Faraday
  use interface_IO_input_star4ve_export
  implicit none
  integer :: iAB 
  character(30) :: char1
  character*400 :: dir_path
  real(8) :: xcac, ycac, zcac
  real(8) :: xcoc, ycoc, zcoc
  real(8) :: emdca,  omefca, psica,  alphca, psi4ca, psif4ca
  real(8) :: bvxdca, bvydca, bvzdca, bvxuca, bvyuca, bvzuca
  real(8) :: hxxdca, hxydca, hxzdca, hyydca, hyzdca, hzzdca
  real(8) :: hxxuca, hxyuca, hxzuca, hyyuca, hyzuca, hzzuca
  real(8) :: hca, preca, rhoca, eneca, epsca
  real(8) :: kxxca, kxyca, kxzca, kyyca, kyzca, kzzca
  real(8) :: vxu, vyu, vzu
  real(8) :: bxcor, bycor, bzcor, bvxufca, bvyufca, bvzufca, psifca, alphfca
  real(8) :: gxx1, gxy1, gxz1, gyy1, gyz1, gzz1, kxx1, kxy1, kxz1, kyy1, kyz1, kzz1
  real(8) :: ome, ber, radi
  real(8) :: va1, vaxd1, vayd1, vazd1, fxd1, fyd1, fzd1, fxyd1, fxzd1, fyzd1
  real(8) :: vaca, vaxdca, vaydca, vazdca, fxdca, fydca, fzdca, fxydca, fxzdca, fyzdca
  real(8) :: utfca, uxfca, uyfca, uzfca
!
  real(8), pointer :: emd(:,:,:), omef(:,:,:), rs(:,:)
  real(8), pointer :: utf(:,:,:)  ,  uxf(:,:,:) ,  uyf(:,:,:) ,  uzf(:,:,:)
  real(8), pointer :: psif(:,:,:), alphf(:,:,:), bvxuf(:,:,:), bvyuf(:,:,:), bvzuf(:,:,:)
  real(8), pointer :: psi(:,:,:)  , alph(:,:,:)
  real(8), pointer :: bvxd(:,:,:) , bvyd(:,:,:) , bvzd(:,:,:) , bvxu(:,:,:) , bvyu(:,:,:), bvzu(:,:,:)
  real(8), pointer :: hxxd(:,:,:) , hxyd(:,:,:) , hxzd(:,:,:) , hyyd(:,:,:) , hyzd(:,:,:), hzzd(:,:,:)
  real(8), pointer :: hxxu(:,:,:) , hxyu(:,:,:) , hxzu(:,:,:) , hyyu(:,:,:) , hyzu(:,:,:), hzzu(:,:,:)
  real(8), pointer :: kxx(:,:,:)  , kxy(:,:,:)  , kxz(:,:,:)  , kyy(:,:,:)  , kyz(:,:,:) , kzz(:,:,:)
  real(8), pointer ::  va(:,:,:)  , vaxd(:,:,:) , vayd(:,:,:) , vazd(:,:,:)
  real(8), pointer :: fxd(:,:,:)  ,  fyd(:,:,:) ,  fzd(:,:,:) , fxyd(:,:,:) , fxzd(:,:,:), fyzd(:,:,:)
!
  gxx1=0.0d0; gxy1=0.0d0; gxz1=0.0d0; gyy1=0.0d0; gyz1=0.0d0; gzz1=0.0d0
  kxx1=0.0d0; kxy1=0.0d0; kxz1=0.0d0; kyy1=0.0d0; kyz1=0.0d0; kzz1=0.0d0
  kxxca=0.0d0; kxyca=0.0d0; kxzca=0.0d0; kyyca=0.0d0; kyzca=0.0d0; kzzca=0.0d0
  vaca=0.0d0;  vaxdca=0.0d0;  vaydca=0.0d0;  vazdca=0.0d0;  
  fxdca=0.0d0; fydca=0.0d0; fzdca=0.0d0; fxydca=0.0d0; fxzdca=0.0d0; fyzdca=0.0d0; 
  fxd1=0.0d0;  fyd1=0.0d0;  fzd1=0.0d0;  fxyd1=0.0d0;  fxzd1=0.0d0; fyzd1=0.0d0;  
  va1=0.0d0;  vaxd1=0.0d0; vayd1=0.0d0; vazd1=0.0d0;
  utfca=0.0d0;  uxfca=0.0d0;  uyfca=0.0d0;  uzfca=0.0d0; 

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
  if (iAB .eq. 1) then
    write(*,*) "Exporting Ai"
  else
    write(*,*) "Exporting Bi" 
  end if

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
  allocate (bvxuf(0:nrf,0:ntf,0:npf))
  allocate (bvyuf(0:nrf,0:ntf,0:npf))
  allocate (bvzuf(0:nrf,0:ntf,0:npf))
  allocate (   rs(0:ntf,0:npf))
  allocate (  utf(0:nrf,0:ntf,0:npf))
  allocate (  uxf(0:nrf,0:ntf,0:npf))
  allocate (  uyf(0:nrf,0:ntf,0:npf))
  allocate (  uzf(0:nrf,0:ntf,0:npf))

  allocate (  psi(0:nrg,0:ntg,0:npg))
  allocate ( alph(0:nrg,0:ntg,0:npg))
  allocate ( bvxd(0:nrg,0:ntg,0:npg))
  allocate ( bvyd(0:nrg,0:ntg,0:npg))
  allocate ( bvzd(0:nrg,0:ntg,0:npg))
  allocate ( bvxu(0:nrg,0:ntg,0:npg))
  allocate ( bvyu(0:nrg,0:ntg,0:npg))
  allocate ( bvzu(0:nrg,0:ntg,0:npg))
  allocate ( hxxd(0:nrg,0:ntg,0:npg))
  allocate ( hxyd(0:nrg,0:ntg,0:npg))
  allocate ( hxzd(0:nrg,0:ntg,0:npg))
  allocate ( hyyd(0:nrg,0:ntg,0:npg))
  allocate ( hyzd(0:nrg,0:ntg,0:npg))
  allocate ( hzzd(0:nrg,0:ntg,0:npg))
  allocate ( hxxu(0:nrg,0:ntg,0:npg))
  allocate ( hxyu(0:nrg,0:ntg,0:npg))
  allocate ( hxzu(0:nrg,0:ntg,0:npg))
  allocate ( hyyu(0:nrg,0:ntg,0:npg))
  allocate ( hyzu(0:nrg,0:ntg,0:npg))
  allocate ( hzzu(0:nrg,0:ntg,0:npg))
  allocate (  kxx(0:nrg,0:ntg,0:npg))
  allocate (  kxy(0:nrg,0:ntg,0:npg))
  allocate (  kxz(0:nrg,0:ntg,0:npg))
  allocate (  kyy(0:nrg,0:ntg,0:npg))
  allocate (  kyz(0:nrg,0:ntg,0:npg))
  allocate (  kzz(0:nrg,0:ntg,0:npg))
  if (iAB .eq. 1) then
    allocate (   va(0:nrg,0:ntg,0:npg))
    allocate ( vaxd(0:nrg,0:ntg,0:npg))
    allocate ( vayd(0:nrg,0:ntg,0:npg))
    allocate ( vazd(0:nrg,0:ntg,0:npg))
    va=0.0d0;    vaxd=0.0d0;  vayd=0.0d0;   vazd=0.0d0;
  else
    allocate (  fxd(0:nrg,0:ntg,0:npg))
    allocate (  fyd(0:nrg,0:ntg,0:npg))
    allocate (  fzd(0:nrg,0:ntg,0:npg))
    allocate ( fxyd(0:nrg,0:ntg,0:npg))
    allocate ( fxzd(0:nrg,0:ntg,0:npg))
    allocate ( fyzd(0:nrg,0:ntg,0:npg))
    fxd=0.0d0;   fyd=0.0d0;   fzd=0.0d0;    fxyd=0.0d0;   fxzd=0.0d0;  fyzd=0.0d0;
  end if

  emd=0.0d0;  rs  =0.0d0;  omef=0.0d0
  utf=0.0d0;  uxf=0.0d0;   uyf=0.0d0;   uzf=0.0d0;
  psif=0.0d0; alphf=0.0d0; bvxuf=0.0d0; bvyuf=0.0d0; bvzuf=0.0d0
  psi=0.0d0;  alph=0.0d0;  bvxu=0.0d0;  bvyu=0.0d0;  bvzu=0.0d0
  bvxd=0.0d0; bvyd=0.0d0;  bvzd=0.0d0
  kxx=0.0d0;  kxy =0.0d0;  kxz =0.0d0;   kyy=0.0d0;   kyz=0.0d0;   kzz=0.0d0
  hxxd=0.0d0; hxyd=0.0d0;  hxzd=0.0d0;  hyyd=0.0d0;  hyzd=0.0d0;  hzzd=0.0d0;
  hxxu=0.0d0; hxyu=0.0d0;  hxzu=0.0d0;  hyyu=0.0d0;  hyzu=0.0d0;  hzzu=0.0d0;

  call IO_input_CF_grav_export(trim(dir_path)//"/rnsgra_3D.las",psi,alph,bvxd,bvyd,bvzd)

  call IO_input_WL_grav_export_hij(trim(dir_path)//"/rnsgra_hij_3D.las",hxxd,hxyd,hxzd,hyyd,hyzd,hzzd)

  call IO_input_grav_export_Kij(trim(dir_path)//"/rnsgra_Kij_3D.las",kxx,kxy,kxz,kyy,kyz,kzz)

  if (iAB .eq. 1) then
    call IO_input_grav_export_Ai(trim(dir_path)//"/rnsEMF_3D.las",va,vaxd,vayd,vazd)
  else
    call IO_input_grav_export_Faraday(trim(dir_path)//"/rnsEMF_faraday_3D.las",fxd,fyd,fzd,fxyd,fxzd,fyzd)
  end if

  call IO_input_CF_star_export(trim(dir_path)//"/rnsflu_3D.las",emd,rs,omef,ome,ber,radi)

  call IO_input_star4ve_export(trim(dir_path)//"/rns4ve_3D.las",utf,uxf,uyf,uzf)

  call invhij_WL_export(hxxd,hxyd,hxzd,hyyd,hyzd,hzzd,hxxu,hxyu,hxzu,hyyu,hyzu,hzzu)

  call index_vec_down2up_export(hxxu,hxyu,hxzu,hyyu,hyzu,hzzu,bvxu,bvyu,bvzu,bvxd,bvyd,bvzd)

  call interpo_gr2fl_metric_CF_export(alph, psi, bvxu, bvyu, bvzu, &
        &    alphf, psif, bvxuf, bvyuf, bvzuf, rs)


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
  call interpo_gr2cgr_4th(bvxu, bvxuca, xcoc, ycoc, zcoc)
  call interpo_gr2cgr_4th(bvyu, bvyuca, xcoc, ycoc, zcoc)
  call interpo_gr2cgr_4th(bvzu, bvzuca, xcoc, ycoc, zcoc)

  call interpo_gr2cgr_4th(hxxd, hxxdca, xcoc, ycoc, zcoc)
  call interpo_gr2cgr_4th(hxyd, hxydca, xcoc, ycoc, zcoc)
  call interpo_gr2cgr_4th(hxzd, hxzdca, xcoc, ycoc, zcoc)
  call interpo_gr2cgr_4th(hyyd, hyydca, xcoc, ycoc, zcoc)
  call interpo_gr2cgr_4th(hyzd, hyzdca, xcoc, ycoc, zcoc)
  call interpo_gr2cgr_4th(hzzd, hzzdca, xcoc, ycoc, zcoc)
  
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
  call interpo_fl2cgr_4th_export(bvxuf, bvxufca , xcoc, ycoc, zcoc, rs)
  call interpo_fl2cgr_4th_export(bvyuf, bvyufca , xcoc, ycoc, zcoc, rs)
  call interpo_fl2cgr_4th_export(bvzuf, bvzufca , xcoc, ycoc, zcoc, rs)

  call interpo_fl2cgr_4th_export(utf, utfca , xcoc, ycoc, zcoc, rs)
  call interpo_fl2cgr_4th_export(uxf, uxfca , xcoc, ycoc, zcoc, rs)
  call interpo_fl2cgr_4th_export(uyf, uyfca , xcoc, ycoc, zcoc, rs)
  call interpo_fl2cgr_4th_export(uzf, uzfca , xcoc, ycoc, zcoc, rs)

  if (iAB .eq. 1) then
    call interpo_gr2cgr_4th(  va,   vaca, xcoc, ycoc, zcoc)
    call interpo_gr2cgr_4th(vaxd, vaxdca, xcoc, ycoc, zcoc)
    call interpo_gr2cgr_4th(vayd, vaydca, xcoc, ycoc, zcoc)
    call interpo_gr2cgr_4th(vazd, vazdca, xcoc, ycoc, zcoc)
  else
    call interpo_gr2cgr_4th(fxd,   fxdca, xcoc, ycoc, zcoc)
    call interpo_gr2cgr_4th(fyd,   fydca, xcoc, ycoc, zcoc)
    call interpo_gr2cgr_4th(fzd,   fzdca, xcoc, ycoc, zcoc)
    call interpo_gr2cgr_4th(fxyd, fxydca, xcoc, ycoc, zcoc)
    call interpo_gr2cgr_4th(fxzd, fxzdca, xcoc, ycoc, zcoc)
    call interpo_gr2cgr_4th(fyzd, fyzdca, xcoc, ycoc, zcoc)
  end if

  bxcor = bvxufca + omefca*(-ycoc)
  bycor = bvyufca + omefca*(xcoc)
  bzcor = bvzufca
  psi4ca = psica**4
  psif4ca = psifca**4

  if (dabs(emdca) > 1.0d-14) then
!    vxu = bxcor/alphfca 
!    vyu = bycor/alphfca
!    vzu = bzcor/alphfca
    vxu = (uxfca/utfca + bvxufca)/alphfca    
    vyu = (uyfca/utfca + bvyufca)/alphfca 
    vzu = (uzfca/utfca + bvzufca)/alphfca 
  else
    emdca=0.0d0
    vxu=0.0d0; vyu=0.0d0; vzu=0.0d0
  end if

  gxx1 = psi4ca*(1.0d0+hxxdca)
  gxy1 = psi4ca*(0.0d0+hxydca)
  gxz1 = psi4ca*(0.0d0+hxzdca)
  gyy1 = psi4ca*(1.0d0+hyydca)
  gyz1 = psi4ca*(0.0d0+hyzdca)
  gzz1 = psi4ca*(1.0d0+hzzdca)

  kxx1 = psi4ca*kxxca/(radi)
  kxy1 = psi4ca*kxyca/(radi)
  kxz1 = psi4ca*kxzca/(radi)
  kyy1 = psi4ca*kyyca/(radi)
  kyz1 = psi4ca*kyzca/(radi)
  kzz1 = psi4ca*kzzca/(radi)

  call peos_q2hprho(emdca, hca, preca, rhoca, eneca)

  epsca = eneca/rhoca - 1.0d0

  if (iAB .eq. 1) then
    !Ax(inx,iny,inz) = vaxdca           ! A_i
    !Ay(inx,iny,inz) = vaydca
    !Az(inx,iny,inz) = vazdca

    va1   = vaca
    vaxd1 = vaxdca
    vayd1 = vaydca
    vazd1 = vazdca
  else
    !Ax(inx,iny,inz) =  fyzdca/radi
    !Ay(inx,iny,inz) = -fxzdca/radi
    !Az(inx,iny,inz) =  fxydca/radi      ! B_i   

    fxd1  = fxdca/radi
    fyd1  = fydca/radi
    fzd1  = fzdca/radi
    fxyd1 = fxydca/radi
    fxzd1 = fxzdca/radi
    fyzd1 = fyzdca/radi
  end if


  write(6,'(a6,e20.12)') "psi  =", psica
  write(6,'(a6,e20.12)') "alph =", alphca
  write(6,'(a6,e20.12)') "Radi =", radi
  write(6,'(a6,e20.12)') "Omeg =", ome/radi
  write(6,'(a6,e20.12)') "emd  =", emdca
  write(6,'(a6,e20.12)') "h    =", hca
  write(6,'(a6,e20.12)') "pre  =", preca
  write(6,'(a6,e20.12)') "rho  =", rhoca
  write(6,'(a6,e20.12)') "ene  =", eneca
  write(6,'(a6,e20.12)') "eps  =", epsca
!
  write(6,'(a18)') "gij at gridpoints:"
  write(6,'(3e20.12)') gxx1, gxy1, gxz1
  write(6,'(3e20.12)') gxy1, gyy1, gyz1
  write(6,'(3e20.12)') gxz1, gyz1, gzz1

  write(6,'(a18)') "Kij at gridpoints:"
  write(6,'(3e20.12)') kxx1, kxy1, kxz1
  write(6,'(3e20.12)') kxy1, kyy1, kyz1
  write(6,'(3e20.12)') kxz1, kyz1, kzz1

  write(6,'(a13)') "v^i Eulerian:"
  write(6,'(a6,e20.12)') "vxu  =", vxu
  write(6,'(a6,e20.12)') "vyu  =", vyu
  write(6,'(a6,e20.12)') "vzu  =", vzu

  if (iAB .eq. 1) then
    write(6,'(a13)') "A_i EMF:"
    write(6,'(a6,e20.12)') "va    =", va1
    write(6,'(a6,e20.12)') "vaxd  =", vaxd1
    write(6,'(a6,e20.12)') "vayd  =", vayd1
    write(6,'(a6,e20.12)') "vazd  =", vazd1
  else
    write(6,'(a13)') "Faraday:"
    write(6,'(a6,e20.12)') "fxd   =", fxd1
    write(6,'(a6,e20.12)') "fyd   =", fyd1
    write(6,'(a6,e20.12)') "fzd   =", fzd1
    write(6,'(a6,e20.12)') "fxyd  =", fxyd1
    write(6,'(a6,e20.12)') "fxzd  =", fxzd1
    write(6,'(a6,e20.12)') "fyzd  =", fyzd1
  end if

  write(6,'(a16)') "Deallocating...."
  deallocate(  emd);  deallocate( omef);  deallocate( psif);  deallocate(alphf);    
  deallocate(bvxuf);  deallocate(bvyuf);  deallocate(bvzuf);  deallocate(   rs);    
  deallocate(  psi);  deallocate( alph);  deallocate( bvxd);  deallocate( bvyd);  
  deallocate( bvzd);  deallocate( bvxu);  deallocate( bvyu);  deallocate( bvzu);
  deallocate( hxxd);  deallocate( hxyd);  deallocate( hxzd);  deallocate( hyyd);  
  deallocate( hyzd);  deallocate( hzzd);  deallocate( hxxu);  deallocate( hxyu);  
  deallocate( hxzu);  deallocate( hyyu);  deallocate( hyzu);  deallocate( hzzu);  
  deallocate(  kxx);  deallocate(  kxy);  deallocate(  kxz);  deallocate(  kyy);  
  deallocate(  kyz);  deallocate(  kzz);  
  deallocate(  utf);  deallocate(  uxf);  deallocate(  uyf);  deallocate(  uzf);
  if (iAB .eq. 1) then
    deallocate(   va);  deallocate( vaxd);  deallocate( vayd);  deallocate( vazd);
  else
    deallocate(  fxd);  deallocate(  fyd);  deallocate(  fzd);
    deallocate( fxyd);  deallocate( fxzd);  deallocate( fyzd);
  end if
!
END PROGRAM coc2cac
