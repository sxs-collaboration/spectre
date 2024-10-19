module def_binary_parameter_mpt
  use phys_constant, only : long, nnmpt
  implicit none
  real(long) :: sepa_(nnmpt), dis_(nnmpt), mass_ratio_(nnmpt)
  real(long) :: sepa_proper_(nnmpt), dis_proper_(nnmpt)
  real(long) :: dis_grav_x_(nnmpt), dis_grav_y_(nnmpt), dis_grav_z_(nnmpt)
end module def_binary_parameter_mpt
