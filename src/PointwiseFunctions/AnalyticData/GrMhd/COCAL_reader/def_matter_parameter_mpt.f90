module def_matter_parameter_mpt
  use phys_constant, only : long
  implicit none
  real(long), pointer :: def_matter_param_real_(:,:)
  character(len=2), pointer :: def_matter_param_char2_(:,:)
  integer, pointer :: def_matter_param_int_(:,:)
end module def_matter_parameter_mpt
