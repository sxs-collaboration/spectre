! ----------------------------------------------------------------------
!  Copy array
! ----------------------------------------------------------------------
module copy_int_array_static_2dto3d_mpt
  use phys_constant, only : long, nnmpt
  implicit none
contains
! - - - - -
! 
subroutine copy_int_arraystatic_2dto3d_mpt(impt,array1,array2,&
                                        &  n1min,n1max,n2min,n2max)
  implicit none
  integer, intent(in)  :: n1min, n1max, n2min, n2max, impt
!  integer :: array1(-2:nnrg,-2:nnrg), array2(-2:nnrg,-2:nnrg,1:nnmpt)
  integer :: array1(n1min:n1max,n2min:n2max)
  integer :: array2(n1min:n1max,n2min:n2max,1:nnmpt)
      array2(n1min:n1max,n2min:n2max,impt) &
  & = array1(n1min:n1max,n2min:n2max)
end subroutine copy_int_arraystatic_2dto3d_mpt
end module copy_int_array_static_2dto3d_mpt
