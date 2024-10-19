subroutine allocate_def_matter_parameter_mpt
  use phys_constant, only : nmpt
  use def_matter_parameter_mpt
  use make_array_2d
  use make_char2_array_2d
  use make_int_array_2d
  implicit none
!
  call alloc_array2d(def_matter_param_real_ , 1, 25, 1, nmpt)
  call alloc_char2_array2d(def_matter_param_char2_ , 1, 5, 1, nmpt)
  call alloc_int_array2d(def_matter_param_int_ , 1, 3, 1, nmpt)
!
end subroutine allocate_def_matter_parameter_mpt
