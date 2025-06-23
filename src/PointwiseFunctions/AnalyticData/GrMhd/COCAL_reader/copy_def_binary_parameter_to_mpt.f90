subroutine copy_def_binary_parameter_to_mpt(impt)
  use def_binary_parameter
  use def_binary_parameter_mpt
  use copy_array_static_0dto1d_mpt
  implicit none
  integer :: impt
!
  call copy_arraystatic_0dto1d_mpt(impt, sepa, sepa_)
  call copy_arraystatic_0dto1d_mpt(impt, dis, dis_)
  call copy_arraystatic_0dto1d_mpt(impt, mass_ratio, mass_ratio_)
!
  call copy_arraystatic_0dto1d_mpt(impt, sepa_proper, sepa_proper_)
  call copy_arraystatic_0dto1d_mpt(impt, dis_proper,  dis_proper_)
  call copy_arraystatic_0dto1d_mpt(impt, dis_grav_x,  dis_grav_x_)
  call copy_arraystatic_0dto1d_mpt(impt, dis_grav_y,  dis_grav_y_)
  call copy_arraystatic_0dto1d_mpt(impt, dis_grav_z,  dis_grav_z_)
!
end subroutine copy_def_binary_parameter_to_mpt
