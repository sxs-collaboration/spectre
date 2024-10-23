module grid_parameter_mpt
  use phys_constant, only : long
  implicit none
  integer, pointer :: grid_param_int_(:,:), surf_param_int_(:,:)
  character(len=2), pointer :: grid_param_char2_(:,:)
  character(len=1), pointer :: grid_param_char1_(:,:)
  real(long), pointer :: grid_param_real_(:,:), surf_param_real_(:,:)
  character(len=1), pointer :: surf_param_char1_(:,:)
end module grid_parameter_mpt
