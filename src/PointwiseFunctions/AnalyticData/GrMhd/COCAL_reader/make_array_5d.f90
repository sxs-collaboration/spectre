! ----------------------------------------------------------------------
!  Allocate array
! ----------------------------------------------------------------------
module make_array_5d
  use phys_constant, only : long
  implicit none
contains
! - - - - -
! 5D array
subroutine alloc_array5d(array,n1min,n1max,n2min,n2max,n3min,n3max,n4min,n4max,n5min,n5max)
  implicit none
  integer,Intent(IN)   :: n1min, n1max, n2min, n2max, n3min, n3max, n4min, n4max, n5min, n5max
  integer              :: status
  Real(long), Pointer  :: array(:,:,:,:,:)
  Allocate(array(n1min:n1max,n2min:n2max,n3min:n3max,n4min:n4max,n5min:n5max),stat=status)
end subroutine alloc_array5d
end module make_array_5d
