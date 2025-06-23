!  trigonometric functions in phi coordinate
!______________________________________________
MODULE trigonometry_grav_phi
  use phys_constant, only  : nnpg, long
  use grid_parameter, only : npg, nlg
  use coordinate_grav_phi, only : phig, hphig
  use make_array_2d
  implicit none
  real(long) ::  sinphig(0:nnpg), cosphig(0:nnpg)
  real(long) :: hsinphig(nnpg),  hcosphig(nnpg)
  real(long), pointer :: sinmpg(:,:), cosmpg(:,:)
  real(long), pointer :: hsinmpg(:,:), hcosmpg(:,:)
contains
! Subroutine
subroutine allocate_trig_grav_mphi
  implicit none
  call alloc_array2d(sinmpg, 0, nlg, 0, npg)
  call alloc_array2d(cosmpg, 0, nlg, 0, npg)
  call alloc_array2d(hsinmpg, 0, nlg, 1, npg)
  call alloc_array2d(hcosmpg, 0, nlg, 1, npg)
end subroutine allocate_trig_grav_mphi
subroutine trig_grav_phi
  IMPLICIT NONE
  INTEGER     ::  ip, nn
  Real(long)  ::  fnn
  DO ip  = 0, npg
    sinphig(ip) = SIN(phig(ip))
    cosphig(ip) = COS(phig(ip))
    DO nn  = 0, nlg
      fnn = Real(nn)
      sinmpg(nn,ip) = SIN(fnn*phig(ip))
      cosmpg(nn,ip) = COS(fnn*phig(ip))
    END DO
  END DO
  DO ip  = 1, npg
    hsinphig(ip) = SIN(hphig(ip))
    hcosphig(ip) = COS(hphig(ip))
    DO nn  = 0, nlg
      fnn = Real(nn)
      hsinmpg(nn,ip) = SIN(fnn*hphig(ip))
      hcosmpg(nn,ip) = COS(fnn*hphig(ip))
    END DO
  END DO
end subroutine trig_grav_phi
end module trigonometry_grav_phi
