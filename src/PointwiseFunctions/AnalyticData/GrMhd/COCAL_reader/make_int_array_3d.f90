! ----------------------------------------------------------------------
!  Allocate 2D array
! ----------------------------------------------------------------------
module make_int_array_3d
  use phys_constant, only : long
  implicit none
contains
! 2D array
subroutine alloc_int_array3d(array,n1min,n1max,n2min,n2max,n3min,n3max)
  implicit none
  integer,Intent(IN)  :: n1min, n1max, n2min, n2max, n3min, n3max
  integer             :: status
  integer, Pointer    :: array(:,:,:)
  Allocate(array(n1min:n1max,n2min:n2max,n3min:n3max),stat=status)
end subroutine alloc_int_array3d
end module make_int_array_3d
