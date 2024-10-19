!  phi_coordinate
!______________________________________________
module coordinate_grav_phi_mpt
  use phys_constant,  only : pi, nnpg, long, nnmpt
  use grid_parameter, only : npg
  implicit none
  Real(long) :: dphig_(nnmpt), dphiginv_(nnmpt)  
  Real(long) :: phig_(0:nnpg,nnmpt), hphig_(nnpg,nnmpt)
! Subroutine
end module coordinate_grav_phi_mpt
