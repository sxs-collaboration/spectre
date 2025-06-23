subroutine read_parameter_mpt(impt)
  use phys_constant, only : long
  use def_matter_parameter, only : emdc, pinx
  use def_quantities, only : restmass_sph, gravmass_sph, &
  &                          MoverR_sph, schwarz_radi_sph
  use grid_parameter
  implicit none
  integer,intent(in)  :: impt
  character(len=1) :: np(5) = (/'1', '2','3', '4', '5'/)
  real(long) :: emdc_ini
  open(1,file='rnspar_mpt'//np(impt)//'.dat',status='old')
  read(1,'(4i5)') nrg, ntg, npg, nlg
  read(1,'(4i5)') nrf, ntf, npf, nlf
  read(1,'(2i5,2(3x,a2))') nrf_deform, nrgin, NS_shape, EQ_point
  read(1,'(1p,3e10.3)') rgin, rgmid, rgout
  read(1,'(/,1i5,2(4x,a1))') iter_max, sw_mass_iter, sw_art_deform
  read(1,'(1p,2e10.3)') conv0_gra, conv_ini
  read(1,'(1p,2e10.3)') conv0_den, conv0_vep
  read(1,'(2(3x,a2),3x,3a1)') indata_type, outdata_type, chrot, chgra, chope
  read(1,'(1p,2e10.3)') eps, mass_eps
  read(1,'(/,2i5)') num_sol_seq, deform_par
  read(1,'(1p,2e14.6)') emdc_ini, pinx
  read(1,'(1p,2e14.6)') restmass_sph, gravmass_sph
  read(1,'(1p,2e14.6)') MoverR_sph, schwarz_radi_sph
  close(1)
  ratio = dble(nrf_deform)/dble(nrf)
  emdc = emdc_ini
  ntgpolp = 0; ntgpolm = ntg; ntgeq = ntg/2; ntgxy = ntg/2
  npgxzp = 0; npgxzm = npg/2; npgyzp = npg/4; npgyzm = 3*(npg/4)
  ntfpolp = 0; ntfpolm = ntf; ntfeq = ntf/2; ntfxy = ntf/2
  npfxzp = 0; npfxzm = npf/2; npfyzp = npf/4; npfyzm = 3*(npf/4)
end subroutine read_parameter_mpt

!Modified routine to allow passing an initial data directory as a
!string. Used for Cactus ID import thorn.
subroutine read_parameter_mpt_cactus(impt, dir_path)
  use phys_constant, only : long
  use def_matter_parameter, only : emdc, pinx
  use def_quantities, only : restmass_sph, gravmass_sph, &
  &                          MoverR_sph, schwarz_radi_sph
  use grid_parameter
  implicit none
  integer,intent(in)  :: impt
  character*400, intent(in) :: dir_path
  character(len=1) :: np(5) = (/'1', '2','3', '4', '5'/)
  real(long) :: emdc_ini
  character*400 :: filepath
  write(*,*) dir_path
  !filepath="/home/fs01/ml2847/SPECTRE/BNS/ID/IRE3.0_K123.6_K123.6_030_M2.8/work_area_BNS/rnspar_mpt2.dat"
  open(1,file=trim(dir_path)//'/'//'rnspar_mpt'//np(impt)//'.dat',status='old')
  !open(1,file=trim(filepath),status='old')
  read(1,'(4i5)') nrg, ntg, npg, nlg
  read(1,'(4i5)') nrf, ntf, npf, nlf
  read(1,'(2i5,2(3x,a2))') nrf_deform, nrgin, NS_shape, EQ_point
  read(1,'(1p,3e10.3)') rgin, rgmid, rgout
  read(1,'(/,1i5,2(4x,a1))') iter_max, sw_mass_iter, sw_art_deform
  read(1,'(1p,2e10.3)') conv0_gra, conv_ini
  read(1,'(1p,2e10.3)') conv0_den, conv0_vep
  read(1,'(2(3x,a2),3x,3a1)') indata_type, outdata_type, chrot, chgra, chope
  read(1,'(1p,2e10.3)') eps, mass_eps
  read(1,'(/,2i5)') num_sol_seq, deform_par
  read(1,'(1p,2e14.6)') emdc_ini, pinx
  read(1,'(1p,2e14.6)') restmass_sph, gravmass_sph
  read(1,'(1p,2e14.6)') MoverR_sph, schwarz_radi_sph
  close(1)
  ratio = dble(nrf_deform)/dble(nrf)
  emdc = emdc_ini
  ntgpolp = 0; ntgpolm = ntg; ntgeq = ntg/2; ntgxy = ntg/2
  npgxzp = 0; npgxzm = npg/2; npgyzp = npg/4; npgyzm = 3*(npg/4)
  ntfpolp = 0; ntfpolm = ntf; ntfeq = ntf/2; ntfxy = ntf/2
  npfxzp = 0; npfxzm = npf/2; npfyzp = npf/4; npfyzm = 3*(npf/4)
end subroutine read_parameter_mpt_cactus

