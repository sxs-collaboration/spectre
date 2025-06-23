! ----------------------------------------------------------------------
!  Allocate array
! ----------------------------------------------------------------------
module make_array_4d
  use phys_constant, only : long
  implicit none
contains
! - - - - -
! 4D array
subroutine alloc_array4d(array,n1min,n1max,n2min,n2max,n3min,n3max,n4min,n4max)
  implicit none
  integer,Intent(IN)   :: n1min, n1max, n2min, n2max, n3min, n3max, n4min, n4max
  integer              :: status
  Real(long), Pointer  :: array(:,:,:,:)
  Allocate(array(n1min:n1max,n2min:n2max,n3min:n3max,n4min:n4max),stat=status)
end subroutine alloc_array4d
end module make_array_4d
