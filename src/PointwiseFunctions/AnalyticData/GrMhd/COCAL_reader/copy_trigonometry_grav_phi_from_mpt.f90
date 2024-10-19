!  trigonometric functions in phi coordinate
!______________________________________________
subroutine copy_trigonometry_grav_phi_from_mpt(impt)
  use phys_constant, only : nnpg
  use grid_parameter, only : npg, nlg
  use trigonometry_grav_phi
  use trigonometry_grav_phi_mpt
  use copy_array_static_2dto1d_mpt
  use copy_array_3dto2d_mpt
  implicit none
  integer :: impt
!
  call copy_arraystatic_2dto1d_mpt(impt, sinphig_, sinphig, 0, nnpg)
  call copy_arraystatic_2dto1d_mpt(impt, cosphig_, cosphig, 0, nnpg)
  call copy_arraystatic_2dto1d_mpt(impt, hsinphig_, hsinphig, 1, nnpg)
  call copy_arraystatic_2dto1d_mpt(impt, hcosphig_, hcosphig, 1, nnpg)
  call copy_array3dto2d_mpt(impt, sinmpg_, sinmpg, 0, nlg, 0, npg)
  call copy_array3dto2d_mpt(impt, cosmpg_, cosmpg, 0, nlg, 0, npg)
  call copy_array3dto2d_mpt(impt, hsinmpg_, hsinmpg, 0, nlg, 1, npg)
  call copy_array3dto2d_mpt(impt, hcosmpg_, hcosmpg, 0, nlg, 1, npg)
!
end subroutine copy_trigonometry_grav_phi_from_mpt
