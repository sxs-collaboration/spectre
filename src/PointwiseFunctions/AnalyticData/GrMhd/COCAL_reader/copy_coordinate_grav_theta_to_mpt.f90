!  theta_coordinate
!______________________________________________
subroutine copy_coordinate_grav_theta_to_mpt(impt)
  use phys_constant, only : nntg 
  use coordinate_grav_theta
  use coordinate_grav_theta_mpt
  use copy_array_static_0dto1d_mpt
  use copy_array_static_1dto2d_mpt
  
  implicit none
  integer :: impt
!
  call copy_arraystatic_0dto1d_mpt(impt, dthg, dthg_)
  call copy_arraystatic_0dto1d_mpt(impt, dthginv, dthginv_)
  call copy_arraystatic_1dto2d_mpt(impt, thg, thg_, 0, nntg)
  call copy_arraystatic_1dto2d_mpt(impt, hthg, hthg_, 1, nntg)
!
end subroutine copy_coordinate_grav_theta_to_mpt
