! r_coordinate for the field
!______________________________________________
subroutine copy_coordinate_grav_r_from_mpt(impt)
  use phys_constant, only : nnrg
  use coordinate_grav_r
  use coordinate_grav_r_mpt
  use copy_array_static_2dto1d_mpt
  implicit none
  integer :: impt
!
  call copy_arraystatic_2dto1d_mpt(impt, rg_, rg, 0, nnrg)
  call copy_arraystatic_2dto1d_mpt(impt, rginv_, rginv, 0, nnrg)
  call copy_arraystatic_2dto1d_mpt(impt, hrg_, hrg, 1, nnrg)
  call copy_arraystatic_2dto1d_mpt(impt, hrginv_, hrginv, 1, nnrg)
  call copy_arraystatic_2dto1d_mpt(impt, drg_, drg, 1, nnrg)
  call copy_arraystatic_2dto1d_mpt(impt, drginv_, drginv, 1, nnrg)
!
end subroutine copy_coordinate_grav_r_from_mpt
