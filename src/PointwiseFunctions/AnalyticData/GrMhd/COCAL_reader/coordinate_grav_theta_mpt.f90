!  theta_coordinate
!______________________________________________
module coordinate_grav_theta_mpt
  use phys_constant,  only : pi, nntg, long, nnmpt
  use grid_parameter, only : ntg 
  implicit none
  Real(long) :: dthg_(nnmpt), dthginv_(nnmpt) 
  Real(long) :: thg_(0:nntg,nnmpt), hthg_(nntg,nnmpt)
end module coordinate_grav_theta_mpt
