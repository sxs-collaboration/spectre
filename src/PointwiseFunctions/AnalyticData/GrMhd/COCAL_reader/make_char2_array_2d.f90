! ----------------------------------------------------------------------
!  Allocate 2D array
! ----------------------------------------------------------------------
module make_char2_array_2d
  use phys_constant, only : long
  implicit none
contains
! 2D array
subroutine alloc_char2_array2d(array,n1min,n1max,n2min,n2max)
  implicit none
  integer,Intent(IN)  :: n1min, n1max, n2min, n2max
  integer             :: status
  character(len=2), Pointer    :: array(:,:)
  Allocate(array(n1min:n1max,n2min:n2max),stat=status)
end subroutine alloc_char2_array2d
end module make_char2_array_2d
