subroutine copy_grid_parameter_from_mpt(impt)
  use grid_parameter_mpt
  use grid_parameter
  use def_quantities
  implicit none
  integer :: i, impt
!
  i=0
  i=i+1; nrg = grid_param_int_(i,impt)
  i=i+1; ntg = grid_param_int_(i,impt)
  i=i+1; npg = grid_param_int_(i,impt)
  i=i+1; nlg = grid_param_int_(i,impt)
  i=i+1; nrf = grid_param_int_(i,impt)
  i=i+1; ntf = grid_param_int_(i,impt)
  i=i+1; npf = grid_param_int_(i,impt)
  i=i+1; nlf = grid_param_int_(i,impt)
  i=i+1; nrf_deform = grid_param_int_(i,impt)
  i=i+1; nrgin      = grid_param_int_(i,impt)
  i=i+1; ntgpolp    = grid_param_int_(i,impt)
  i=i+1; ntgpolm    = grid_param_int_(i,impt)
  i=i+1; ntgeq  = grid_param_int_(i,impt)
  i=i+1; ntgxy  = grid_param_int_(i,impt)
  i=i+1; npgxzp = grid_param_int_(i,impt)
  i=i+1; npgxzm = grid_param_int_(i,impt)
  i=i+1; npgyzp = grid_param_int_(i,impt)
  i=i+1; npgyzm = grid_param_int_(i,impt)
  i=i+1; ntfpolp = grid_param_int_(i,impt)
  i=i+1; ntfpolm = grid_param_int_(i,impt)
  i=i+1; ntfeq   = grid_param_int_(i,impt)
  i=i+1; ntfxy   = grid_param_int_(i,impt)
  i=i+1; npfxzp  = grid_param_int_(i,impt)
  i=i+1; npfxzm  = grid_param_int_(i,impt)
  i=i+1; npfyzp  = grid_param_int_(i,impt)
  i=i+1; npfyzm  = grid_param_int_(i,impt)
  i=i+1; iter_max    = grid_param_int_(i,impt)
  i=i+1; num_sol_seq = grid_param_int_(i,impt)
  i=i+1; deform_par  = grid_param_int_(i,impt)
!
  i=0
  i=i+1; indata_type = grid_param_char2_(i,impt)
  i=i+1; outdata_type = grid_param_char2_(i,impt)
  i=i+1; NS_shape = grid_param_char2_(i,impt)
  i=i+1; EQ_point = grid_param_char2_(i,impt)
!
  i=0
  i=i+1; chrot = grid_param_char1_(i,impt)
  i=i+1; chgra = grid_param_char1_(i,impt)
  i=i+1; chope = grid_param_char1_(i,impt)
  i=i+1; sw_mass_iter = grid_param_char1_(i,impt)
  i=i+1; sw_art_deform = grid_param_char1_(i,impt)
!
  i=0
  i=i+1; rgin  = grid_param_real_(i,impt)
  i=i+1; rgmid = grid_param_real_(i,impt)
  i=i+1; rgout = grid_param_real_(i,impt)
  i=i+1; ratio = grid_param_real_(i,impt)
  i=i+1; conv_gra = grid_param_real_(i,impt)
  i=i+1; conv_den = grid_param_real_(i,impt)
  i=i+1; conv_vep = grid_param_real_(i,impt)
  i=i+1; conv_ini = grid_param_real_(i,impt)
  i=i+1; conv0_gra = grid_param_real_(i,impt)
  i=i+1; conv0_den = grid_param_real_(i,impt)
  i=i+1; conv0_vep = grid_param_real_(i,impt)
  i=i+1; eps       = grid_param_real_(i,impt)
  i=i+1; mass_eps  = grid_param_real_(i,impt)
! 
  i=i+1; restmass_sph     = grid_param_real_(i,impt)
  i=i+1; gravmass_sph     = grid_param_real_(i,impt)
  i=i+1; MoverR_sph       = grid_param_real_(i,impt)
  i=i+1; schwarz_radi_sph = grid_param_real_(i,impt)
!
  i=0
  i=i+1; nrg_1    = surf_param_int_(i,impt)
  i=i+1; sw_sepa  = surf_param_int_(i,impt)
  i=i+1; sw_quant = surf_param_int_(i,impt)
  i=i+1; sw_spin  = surf_param_int_(i,impt)
!
  i=0
  i=i+1; r_surf      = surf_param_real_(i,impt)
  i=i+1; target_sepa = surf_param_real_(i,impt)
  i=i+1; target_qt   = surf_param_real_(i,impt)
  i=i+1; target_sx   = surf_param_real_(i,impt)
  i=i+1; target_sy   = surf_param_real_(i,impt)
  i=i+1; target_sz   = surf_param_real_(i,impt)
!
  i=0
  i=i+1; sw_eos = surf_param_char1_(i,impt)
!
end subroutine copy_grid_parameter_from_mpt
