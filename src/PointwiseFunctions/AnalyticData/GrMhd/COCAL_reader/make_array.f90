! ----------------------------------------------------------------------
!  Allocate array
! ----------------------------------------------------------------------
module make_array
  use phys_constant, only : long
  implicit none
contains
! - - - - -
! 1D array
subroutine make_array1(array,n1,stn1)
  implicit none
  integer,Intent(IN)  :: n1
  integer,optional    :: stn1
  integer             :: status
  Real(long), pointer :: array(:)
  if (present(stn1)) then 
    Allocate(array(stn1:n1),stat=status)
  else
    Allocate(array(0:n1),stat=status)
  end if
end subroutine make_array1
! - - - - -
! 2D array
subroutine make_array2(array,n1,n2)
  implicit none
  integer,Intent(IN)  :: n1, n2
  integer             :: status
  Real(long), Pointer :: array(:,:)
  Allocate(array(0:n1,0:n2),stat=status)
end subroutine make_array2
! - - - - -
! 3D array
subroutine make_array3(array,n1,n2,n3)
  implicit none
  integer,Intent(IN)   :: n1, n2, n3
  integer              :: status
  Real(long), Pointer  :: array(:,:,:)
  Allocate(array(0:n1,0:n2,0:n3),stat=status)
end subroutine make_array3
end module make_array
