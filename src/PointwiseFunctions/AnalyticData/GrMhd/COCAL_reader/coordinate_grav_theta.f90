!  theta_coordinate
!______________________________________________
module coordinate_grav_theta
  use phys_constant,  only : pi, nntg, long
  use grid_parameter, only : ntg 
  implicit none
  Real(long) :: dthg, dthginv 
  Real(long) :: thg(0:nntg), hthg(nntg)
contains
subroutine grid_theta
  implicit none
  Integer  ::  it
  dthg  = pi/REAL(ntg)
  dthginv=1.0e0/dthg
  thg(0) = 0.0d0
  do it = 1, ntg
    thg(it) = Real(it)*dthg
    hthg(it) = 0.5d0*(thg(it) + thg(it-1))
  end do
end subroutine grid_theta
end module coordinate_grav_theta
