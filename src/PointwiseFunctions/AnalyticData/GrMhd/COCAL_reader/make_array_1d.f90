! ----------------------------------------------------------------------
!  Allocate 1D array
! ----------------------------------------------------------------------
module make_array_1d
  use phys_constant, only : long
  implicit none
contains
! - - - - -
! 1D array
subroutine alloc_array1d(array,n1min,n1max)
  implicit none
  integer,Intent(IN)  :: n1min, n1max
  integer             :: status
  Real(long), pointer :: array(:)
  Allocate(array(n1min:n1max),stat=status)
end subroutine alloc_array1d
end module make_array_1d
