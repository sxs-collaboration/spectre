subroutine allocate_grid_parameter_binary_excision_mpt
  use phys_constant, only : nmpt
  use grid_parameter_binary_excision_mpt
  use make_int_array_2d
  use make_array_2d
  implicit none
!
  call alloc_int_array2d(grid_param_bin_ex_int_ , 1, 10, 1, nmpt)
  call alloc_array2d(grid_param_bin_ex_real_ , 1, 10, 1, nmpt)
!
end subroutine allocate_grid_parameter_binary_excision_mpt
