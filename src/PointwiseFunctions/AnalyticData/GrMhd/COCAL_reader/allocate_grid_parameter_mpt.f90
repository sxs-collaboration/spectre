subroutine allocate_grid_parameter_mpt
  use phys_constant, only : nmpt
  use grid_parameter_mpt
  use make_int_array_2d
  use make_char2_array_2d
  use make_char1_array_2d
  use make_array_2d
  implicit none
!  
  call alloc_int_array2d(grid_param_int_     , 1, 39, 1, nmpt)
  call alloc_char2_array2d(grid_param_char2_ , 1, 10, 1, nmpt)
  call alloc_char1_array2d(grid_param_char1_ , 1, 10, 1, nmpt)
  call alloc_array2d(grid_param_real_        , 1, 20, 1, nmpt)
!
  call alloc_int_array2d(surf_param_int_     , 1, 10, 1, nmpt)
  call alloc_array2d(surf_param_real_        , 1, 10, 1, nmpt)
  call alloc_char1_array2d(surf_param_char1_ , 1, 10, 1, nmpt)
!
end subroutine allocate_grid_parameter_mpt
