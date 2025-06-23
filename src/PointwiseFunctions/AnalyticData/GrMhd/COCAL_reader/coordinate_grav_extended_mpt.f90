! extended coordinate for the field
!______________________________________________
module coordinate_grav_extended_mpt
  use phys_constant, only : nnrg, nntg, nnpg, long, nnmpt
  implicit none
  real(long) ::   rgex_(-2:nnrg+2,nnmpt), &
  &              thgex_(-2:nntg+2,nnmpt), &
  &             phigex_(-2:nnpg+2,nnmpt)
  integer :: irgex_r_(-2:nnrg+2,nnmpt), &
  &          itgex_r_(0:nntg,-2:nnrg+2,nnmpt), &
  &          ipgex_r_(0:nnpg,-2:nnrg+2,nnmpt)
  integer :: itgex_th_(-2:nntg+2,nnmpt), &
  &          ipgex_th_(0:nnpg,-2:nntg+2,nnmpt)
  integer :: ipgex_phi_(-2:nnpg+2,nnmpt)
!
  real(long) :: hrgex_(-2:nnrg+2,nnmpt), &
  &             hthgex_(-2:nntg+2,nnmpt), &
  &             hphigex_(-2:nnpg+2,nnmpt)
  integer :: irgex_hr_(-2:nnrg+2,nnmpt), &
  &          itgex_hr_(1:nntg,-2:nnrg+2,nnmpt), &
  &          ipgex_hr_(1:nnpg,-2:nnrg+2,nnmpt)
  integer :: itgex_hth_(-2:nntg+2,nnmpt), &
  &          ipgex_hth_(1:nnpg,-2:nntg+2,nnmpt)
  integer :: ipgex_hphi_(-2:nnpg+2,nnmpt)
end module coordinate_grav_extended_mpt
