module grid_parameter_binary_excision_mpt
  use phys_constant, only : long
  implicit none
  integer, pointer :: grid_param_bin_ex_int_(:,:)
  real(long), pointer :: grid_param_bin_ex_real_(:,:)
end module grid_parameter_binary_excision_mpt
