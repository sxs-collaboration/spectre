function lagint_2nd(x,y,v)
  use phys_constant, only : long
  implicit none
  real(long) :: lagint_2nd
  real(long) :: x(2),y(2), v
  real(long) :: dx12, dx21
  real(long) :: xv1, xv2, wex1, wex2
!
      dx12 = x(1) - x(2)
      dx21 = - dx12
      xv1 = v - x(1)
      xv2 = v - x(2)
      wex1 = xv2/dx12
      wex2 = xv1/dx21
!
      lagint_2nd = wex1*y(1) + wex2*y(2)
!
end function lagint_2nd
