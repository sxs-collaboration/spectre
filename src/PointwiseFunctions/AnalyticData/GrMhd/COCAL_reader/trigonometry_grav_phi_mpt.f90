!  trigonometric functions in phi coordinate
!______________________________________________
MODULE trigonometry_grav_phi_mpt
  use phys_constant, only  : nnpg, long, nnmpt
  use grid_parameter, only : npg, nlg
  use make_array_2d
  implicit none
  real(long) ::  sinphig_(0:nnpg,nnmpt), cosphig_(0:nnpg,nnmpt)
  real(long) :: hsinphig_(nnpg,nnmpt),  hcosphig_(nnpg,nnmpt)
  real(long), pointer :: sinmpg_(:,:,:), cosmpg_(:,:,:)
  real(long), pointer :: hsinmpg_(:,:,:), hcosmpg_(:,:,:)
end module trigonometry_grav_phi_mpt
