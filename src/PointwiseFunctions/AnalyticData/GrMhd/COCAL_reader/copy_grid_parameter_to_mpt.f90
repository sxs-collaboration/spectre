subroutine copy_grid_parameter_to_mpt(impt)
  use grid_parameter
  use grid_parameter_mpt
  implicit none
  integer :: i, impt
!  
  i=0
  i=i+1; grid_param_int_(i,impt) = nrg 
  i=i+1; grid_param_int_(i,impt) = ntg
  i=i+1; grid_param_int_(i,impt) = npg
  i=i+1; grid_param_int_(i,impt) = nlg
  i=i+1; grid_param_int_(i,impt) = nrf
  i=i+1; grid_param_int_(i,impt) = ntf
  i=i+1; grid_param_int_(i,impt) = npf
  i=i+1; grid_param_int_(i,impt) = nlf
  i=i+1; grid_param_int_(i,impt) = nrf_deform
  i=i+1; grid_param_int_(i,impt) = nrgin
  i=i+1; grid_param_int_(i,impt) = ntgpolp
  i=i+1; grid_param_int_(i,impt) = ntgpolm
  i=i+1; grid_param_int_(i,impt) = ntgeq
  i=i+1; grid_param_int_(i,impt) = ntgxy
  i=i+1; grid_param_int_(i,impt) = npgxzp
  i=i+1; grid_param_int_(i,impt) = npgxzm
  i=i+1; grid_param_int_(i,impt) = npgyzp
  i=i+1; grid_param_int_(i,impt) = npgyzm
  i=i+1; grid_param_int_(i,impt) = ntfpolp
  i=i+1; grid_param_int_(i,impt) = ntfpolm
  i=i+1; grid_param_int_(i,impt) = ntfeq
  i=i+1; grid_param_int_(i,impt) = ntfxy
  i=i+1; grid_param_int_(i,impt) = npfxzp
  i=i+1; grid_param_int_(i,impt) = npfxzm
  i=i+1; grid_param_int_(i,impt) = npfyzp
  i=i+1; grid_param_int_(i,impt) = npfyzm
  i=i+1; grid_param_int_(i,impt) = iter_max
  i=i+1; grid_param_int_(i,impt) = num_sol_seq
  i=i+1; grid_param_int_(i,impt) = deform_par
!
  i=0
  i=i+1; grid_param_char2_(i,impt) = indata_type
  i=i+1; grid_param_char2_(i,impt) = outdata_type
  i=i+1; grid_param_char2_(i,impt) = NS_shape
  i=i+1; grid_param_char2_(i,impt) = EQ_point
!
  i=0
  i=i+1; grid_param_char1_(i,impt) = chrot
  i=i+1; grid_param_char1_(i,impt) = chgra
  i=i+1; grid_param_char1_(i,impt) = chope
  i=i+1; grid_param_char1_(i,impt) = sw_mass_iter
  i=i+1; grid_param_char1_(i,impt) = sw_art_deform
!
  i=0
  i=i+1; grid_param_real_(i,impt) = rgin
  i=i+1; grid_param_real_(i,impt) = rgmid
  i=i+1; grid_param_real_(i,impt) = rgout
  i=i+1; grid_param_real_(i,impt) = ratio
  i=i+1; grid_param_real_(i,impt) = conv_gra
  i=i+1; grid_param_real_(i,impt) = conv_den
  i=i+1; grid_param_real_(i,impt) = conv_vep
  i=i+1; grid_param_real_(i,impt) = conv_ini
  i=i+1; grid_param_real_(i,impt) = conv0_gra
  i=i+1; grid_param_real_(i,impt) = conv0_den
  i=i+1; grid_param_real_(i,impt) = conv0_vep
  i=i+1; grid_param_real_(i,impt) = eps
  i=i+1; grid_param_real_(i,impt) = mass_eps
!
  i=i+1; grid_param_real_(i,impt) = restmass_sph
  i=i+1; grid_param_real_(i,impt) = gravmass_sph
  i=i+1; grid_param_real_(i,impt) = MoverR_sph
  i=i+1; grid_param_real_(i,impt) = schwarz_radi_sph
!
  i=0
  i=i+1; surf_param_int_(i,impt) = nrg_1         ! 1
  i=i+1; surf_param_int_(i,impt) = sw_sepa       ! 2
  i=i+1; surf_param_int_(i,impt) = sw_quant      ! 3
  i=i+1; surf_param_int_(i,impt) = sw_spin       ! 4
!
  i=0
  i=i+1; surf_param_real_(i,impt) = r_surf       ! 1
  i=i+1; surf_param_real_(i,impt) = target_sepa  ! 2
  i=i+1; surf_param_real_(i,impt) = target_qt    ! 3
  i=i+1; surf_param_real_(i,impt) = target_sx    ! 4
  i=i+1; surf_param_real_(i,impt) = target_sy    ! 5
  i=i+1; surf_param_real_(i,impt) = target_sz    ! 6
!
  i=0
  i=i+1; surf_param_char1_(i,impt) = sw_eos
!
end subroutine copy_grid_parameter_to_mpt
