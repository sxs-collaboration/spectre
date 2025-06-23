subroutine copy_grid_parameter_binary_excision_from_mpt(impt)
  use grid_parameter_binary_excision_mpt
  use grid_parameter_binary_excision
  implicit none
  integer :: i, impt
!  
  i=0
  i=i+1; ex_nrg  = grid_param_bin_ex_int_(i,impt)
  i=i+1; ex_ndis = grid_param_bin_ex_int_(i,impt)

  i=0
  i=i+1; ex_radius = grid_param_bin_ex_real_(i,impt)
  i=i+1; ex_rgin   = grid_param_bin_ex_real_(i,impt)
  i=i+1; ex_rgmid  = grid_param_bin_ex_real_(i,impt)
  i=i+1; ex_rgout  = grid_param_bin_ex_real_(i,impt)
  
end subroutine copy_grid_parameter_binary_excision_from_mpt
