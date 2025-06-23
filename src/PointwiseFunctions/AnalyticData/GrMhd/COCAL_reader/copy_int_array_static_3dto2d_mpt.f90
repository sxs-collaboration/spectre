! ----------------------------------------------------------------------
!  Copy array
! ----------------------------------------------------------------------
module copy_int_array_static_3dto2d_mpt
  use phys_constant, only : long, nnmpt
  implicit none
contains
! - - - - -
! 
subroutine copy_int_arraystatic_3dto2d_mpt(impt,array1,array2,&
                                        &  n1min,n1max,n2min,n2max)
  implicit none
  integer, intent(in)  :: n1min, n1max, n2min, n2max, impt
!  integer :: array1(-2:nnrg,-2:nnrg,1:nnmpt), array2(-2:nnrg,-2:nnrg)
  integer :: array1(n1min:n1max,n2min:n2max,1:nnmpt)
  integer :: array2(n1min:n1max,n2min:n2max)
      array2(n1min:n1max,n2min:n2max) &
  & = array1(n1min:n1max,n2min:n2max,impt)
end subroutine copy_int_arraystatic_3dto2d_mpt
end module copy_int_array_static_3dto2d_mpt
