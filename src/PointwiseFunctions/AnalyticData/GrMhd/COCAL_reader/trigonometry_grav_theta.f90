!  trigonometric functions in theta coordinate
!______________________________________________
MODULE trigonometry_grav_theta
  use phys_constant, only : nntg, long
  use grid_parameter, only : ntg 
  use coordinate_grav_theta, only : thg, hthg
  implicit none
  real(long) :: sinthg(0:nntg),costhg(0:nntg)
  real(long) :: cosecthg(0:nntg),cotanthg(0:nntg)
  real(long) :: hsinthg(nntg), hcosthg(nntg)
  real(long) :: hcosecthg(nntg), hcotanthg(nntg)
contains
! --- calculate sine ,cosine and tange on grids of polar angle theta. ---
! --- notice ; tange diverge to infinity when agr=pi/2
!
subroutine trig_grav_theta
IMPLICIT NONE
INTEGER  ::  it
  DO it  = 0, ntg
    sinthg(it) = SIN(thg(it))
    costhg(it) = COS(thg(it))
  END DO
  cosecthg( 0) = 0.0e0  ! should be +inf, should not be used in the code
  cotanthg( 0) = 0.0e0  ! should be +inf, should not be used in the code
  DO it = 1, ntg - 1
    cosecthg(it) = 1.0e0/SIN(thg(it))
    cotanthg(it) = 1.0e0/TAN(thg(it))
  END DO
  cosecthg(ntg) = 1.0e0  ! should be +inf, should not be used in the code
  cotanthg(ntg) = 0.0e0  ! should be -inf, should not be used in the code
!
  DO it = 1, ntg
    hsinthg(it) = SIN(hthg(it))
    hcosthg(it) = COS(hthg(it))
  END DO
  DO it = 1, ntg
    hcosecthg(it) = 1.0e0/SIN(hthg(it))
    hcotanthg(it) = 1.0e0/TAN(hthg(it))
  END DO
!
end subroutine trig_grav_theta
end module trigonometry_grav_theta
