!  trigonometric functions in theta coordinate
!______________________________________________
MODULE trigonometry_grav_theta_mpt
  use phys_constant, only : nntg, long,nnmpt
  use grid_parameter, only : ntg 
  use coordinate_grav_theta, only : thg, hthg
  implicit none
  real(long) :: sinthg_(0:nntg,nnmpt),costhg_(0:nntg,nnmpt)
  real(long) :: cosecthg_(0:nntg,nnmpt),cotanthg_(0:nntg,nnmpt)
  real(long) :: hsinthg_(nntg,nnmpt), hcosthg_(nntg,nnmpt)
  real(long) :: hcosecthg_(nntg,nnmpt), hcotanthg_(nntg,nnmpt)
end module trigonometry_grav_theta_mpt
