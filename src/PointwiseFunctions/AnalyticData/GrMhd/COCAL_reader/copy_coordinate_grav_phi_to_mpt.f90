!  phi_coordinate
!______________________________________________
subroutine copy_coordinate_grav_phi_to_mpt(impt)
  use phys_constant, only : nnpg
  use coordinate_grav_phi
  use coordinate_grav_phi_mpt
  use copy_array_static_0dto1d_mpt
  use copy_array_static_1dto2d_mpt
  implicit none
  integer :: impt
!
  call copy_arraystatic_0dto1d_mpt(impt, dphig, dphig_)
  call copy_arraystatic_0dto1d_mpt(impt, dphiginv, dphiginv_)
  call copy_arraystatic_1dto2d_mpt(impt, phig, phig_, 0, nnpg)
  call copy_arraystatic_1dto2d_mpt(impt, hphig, hphig_, 1, nnpg)
!
! Subroutine
end subroutine copy_coordinate_grav_phi_to_mpt
