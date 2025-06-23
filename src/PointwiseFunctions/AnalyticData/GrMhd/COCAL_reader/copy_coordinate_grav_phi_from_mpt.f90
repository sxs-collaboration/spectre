!  phi_coordinate
!______________________________________________
subroutine copy_coordinate_grav_phi_from_mpt(impt)
  use phys_constant, only : nnpg
  use coordinate_grav_phi
  use coordinate_grav_phi_mpt
  use copy_array_static_1dto0d_mpt
  use copy_array_static_2dto1d_mpt
  implicit none
  integer :: impt
!
  call copy_arraystatic_1dto0d_mpt(impt, dphig_, dphig)
  call copy_arraystatic_1dto0d_mpt(impt, dphiginv_, dphiginv)
  call copy_arraystatic_2dto1d_mpt(impt, phig_, phig, 0, nnpg)
  call copy_arraystatic_2dto1d_mpt(impt, hphig_, hphig, 1, nnpg)
!
! Subroutine
end subroutine copy_coordinate_grav_phi_from_mpt
