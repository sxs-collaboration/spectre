! ----------------------------------------------------------------------
!  Allocate array
! ----------------------------------------------------------------------
module make_array_3d
  use phys_constant, only : long
  implicit none
contains
! - - - - -
! 3D array
subroutine alloc_array3d(array,n1min,n1max,n2min,n2max,n3min,n3max)
  implicit none
  integer,Intent(IN)   :: n1min, n1max, n2min, n2max, n3min, n3max
  integer              :: status
  Real(long), Pointer  :: array(:,:,:)
  Allocate(array(n1min:n1max,n2min:n2max,n3min:n3max),stat=status)
  array = 0.0d0
end subroutine alloc_array3d
end module make_array_3d
