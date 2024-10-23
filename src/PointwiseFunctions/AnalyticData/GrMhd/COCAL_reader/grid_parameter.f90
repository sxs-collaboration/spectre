module grid_parameter
  use phys_constant, only : long
  use def_matter_parameter, only : emdc, pinx
  use def_quantities, only : restmass_sph, gravmass_sph, &
  &                          MoverR_sph, schwarz_radi_sph
  implicit none
  integer :: nrg, ntg, npg        ! GR coordinate
  Integer :: nlg                  ! maximum multipole for Legendre
  integer :: nrf, ntf, npf        ! Fluid coordinate
  Integer :: nlf                  ! maximum multipole for Legendre
  Integer :: nrf_deform, nrgin
  integer :: ntgpolp, ntgpolm, ntgeq, ntgxy, npgxzp, npgxzm, npgyzp, npgyzm
  integer :: ntfpolp, ntfpolm, ntfeq, ntfxy, npfxzp, npfxzm, npfyzp, npfyzm
  integer :: iter_max, num_sol_seq, deform_par
  character(2) :: indata_type, outdata_type, NS_shape, EQ_point
  character(1) :: chrot, chgra, chope, sw_mass_iter, sw_art_deform
  real(long) :: rgin, rgmid, rgout, ratio
  real(long) :: conv_gra, conv_den, conv_vep, conv_ini
  real(long) :: conv0_gra, conv0_den, conv0_vep
  real(long) :: eps, mass_eps
  integer    :: nrg_1              ! number of intervals in [0,1]
  real(long) :: r_surf             ! r_surf=rg(nrf)
  character(1) :: sw_L1_iter, sw_eos
  integer    :: sw_sepa, sw_quant, sw_spin
  real(long) :: target_sepa, target_qt, target_sx, target_sy, target_sz
contains
subroutine read_parameter
  implicit none
  real(long) :: emdc_ini
  open(1,file='rnspar.dat',status='old')
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
end subroutine read_parameter

!Modified routine to allow passing an initial data directory as a
!string. Used for Cactus ID import thorn.
subroutine read_parameter_cactus(dir_path)
  implicit none
  character*400, intent(in) :: dir_path
  real(long) :: emdc_ini
  open(1,file=trim(dir_path)//'/'//'rnspar.dat',status='old')
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
end subroutine read_parameter_cactus

end module grid_parameter
