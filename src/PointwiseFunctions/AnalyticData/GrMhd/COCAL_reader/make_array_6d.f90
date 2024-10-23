! ----------------------------------------------------------------------
!  Allocate array
! ----------------------------------------------------------------------
module make_array_6d
  use phys_constant, only : long
  implicit none
contains
! - - - - -
! 6D array
subroutine alloc_array6d(array,n1min,n1max,n2min,n2max,n3min,n3max, &
                         &     n4min,n4max,n5min,n5max,n6min,n6max)
  implicit none
  integer,Intent(IN)   :: n1min, n1max, n2min, n2max, n3min, n3max, &
  &                       n4min, n4max, n5min, n5max, n6min, n6max
  integer              :: status
  Real(long), Pointer  :: array(:,:,:,:,:,:)
  Allocate(array(n1min:n1max,n2min:n2max,n3min:n3max, &
  &              n4min:n4max,n5min:n5max,n6min:n6max),stat=status)
end subroutine alloc_array6d
end module make_array_6d
