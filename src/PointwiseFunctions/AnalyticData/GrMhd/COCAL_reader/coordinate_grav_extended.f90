! extended coordinate for the field
!______________________________________________
module coordinate_grav_extended
  use phys_constant, only : nnrg, nntg, nnpg, long
  use coordinate_grav_r, only : rg, hrg
  use coordinate_grav_theta, only : thg, hthg
  use coordinate_grav_phi, only : phig, hphig
  use grid_parameter, only : nrg, ntg, npg
  implicit none
  real(long) :: rgex(-2:nnrg+2), &
  &             thgex(-2:nntg+2), & 
  &             phigex(-2:nnpg+2)
  integer :: irgex_r(-2:nnrg+2), &
  &          itgex_r(0:nntg,-2:nnrg+2), &
  &          ipgex_r(0:nnpg,-2:nnrg+2)
  integer :: itgex_th(-2:nntg+2), &
  &          ipgex_th(0:nnpg,-2:nntg+2)
  integer :: ipgex_phi(-2:nnpg+2)
!
  real(long) :: hrgex(-2:nnrg+2), &
  &             hthgex(-2:nntg+2), &
  &             hphigex(-2:nnpg+2)
  integer :: irgex_hr(-2:nnrg+2), &
  &          itgex_hr(1:nntg,-2:nnrg+2), &
  &          ipgex_hr(1:nnpg,-2:nnrg+2)
  integer :: itgex_hth(-2:nntg+2), &
  &          ipgex_hth(1:nnpg,-2:nntg+2)
  integer :: ipgex_hphi(-2:nnpg+2)
!
contains
subroutine grid_extended
  implicit none
  integer  :: irg, itg, ipg
  rgex(0:nrg) = rg(0:nrg)
  thgex(0:ntg) = thg(0:ntg)
  phigex(0:npg) = phig(0:npg)
  rgex(-1) = rg(0) - (rg(1) - rg(0))
  rgex(-2) = rg(0) - (rg(2) - rg(0))
  rgex(nrg+1) = rg(nrg) + (rg(nrg) - rg(nrg-1))
  rgex(nrg+2) = rg(nrg) + 2.0d0*(rg(nrg) - rg(nrg-1))
  thgex(-1) = - thg(1)
  thgex(-2) = - thg(2)
  thgex(ntg+1) =  2.0d0*thg(ntg) - thg(ntg-1)
  thgex(ntg+2) =  2.0d0*thg(ntg) - thg(ntg-2)
  phigex(-1) = phig(npg-1) - phig(npg)
  phigex(-2) = phig(npg-2) - phig(npg)
  phigex(npg+1) =  phig(npg) + phig(1)
  phigex(npg+2) =  phig(npg) + phig(2)
!
  do irg = -2, nrg
    if (irg.ge.0.and.irg.le.nrg) irgex_r(irg) = irg
    if (irg.le.-1)    irgex_r(irg) = iabs(irg)
  end do
  do itg = -2, ntg + 2 
    if (itg.ge.0.and.itg.le.ntg) itgex_th(itg) = itg
    if (itg.le.-1)    itgex_th(itg) = iabs(itg)
    if (itg.ge.ntg+1) itgex_th(itg) = 2*ntg - itg
  end do
  do ipg = -2, npg + 2
    if (ipg.ge.0.and.ipg.le.npg) ipgex_phi(ipg) = ipg
    if (ipg.le.-1)    ipgex_phi(ipg) = npg + ipg
    if (ipg.ge.npg+1) ipgex_phi(ipg) = ipg - npg
  end do
!
  do irg = -2, nrg
    do itg = 0, ntg
      if (irg.ge. 0) itgex_r(itg,irg) = itg
      if (irg.le.-1) itgex_r(itg,irg) = ntg - itg
    end do
  end do
  do irg = -2, nrg
    do ipg = 0, npg
      if (irg.ge. 0) ipgex_r(ipg,irg) = ipg
      if (irg.le.-1) ipgex_r(ipg,irg) = mod(ipg + npg/2,npg)
    end do
  end do
  do itg = -2, ntg + 2 
    do ipg = 0, npg
      if (itg.ge.0.and.itg.le.ntg  ) ipgex_th(ipg,itg) = ipg
      if (itg.le.-1.or.itg.ge.ntg+1) ipgex_th(ipg,itg) = mod(ipg + npg/2,npg)
    end do
  end do
!
! midpoints
!
  hrgex(1:nrg) = hrg(1:nrg)
  hthgex(1:ntg) = hthg(1:ntg)
  hphigex(1:npg) = hphig(1:npg)
  hrgex(0) =  rg(0) - hrg(1)
  hrgex(-1) = rg(0) - hrg(2)
  hrgex(-2) = rg(0) - hrg(3)
  hrgex(nrg+1) = 0.5d0*(rg(nrg+1) + rg(nrg))
  hrgex(nrg+2) = 0.5d0*(rg(nrg+2) + rg(nrg+1))
  hthgex(0) = - hthg(1)
  hthgex(-1) = - hthg(2)
  hthgex(-2) = - hthg(3)
  hthgex(ntg+1) =  2.0d0*hthg(ntg) - hthg(ntg-1)
  hthgex(ntg+2) =  2.0d0*hthg(ntg) - hthg(ntg-2)
  hphigex(0) = - hphig(1)
  hphigex(-1) = - hphig(2)
  hphigex(-2) = - hphig(3)
  hphigex(npg+1) =  hphig(npg) + phig(1)
  hphigex(npg+2) =  hphig(npg) + phig(2)
!
  do irg = -2, nrg
    if (irg.ge.1) irgex_hr(irg) = irg
    if (irg.le.0) irgex_hr(irg) = iabs(irg) + 1
  end do
  do irg = -2, nrg
    do itg = 1, ntg
      if (irg.ge.1) itgex_hr(itg,irg) = itg
      if (irg.le.0) itgex_hr(itg,irg) = ntg - itg + 1
    end do
  end do
  do irg = -2, nrg
    do ipg = 1, npg
      if (irg.ge.1) ipgex_hr(ipg,irg) = ipg
      if (irg.le.0) ipgex_hr(ipg,irg) = mod(ipg + npg/2,npg)
    end do
  end do
!
  do itg = -2, ntg + 2 
    if (itg.ge.1.and.itg.le.ntg) itgex_hth(itg) = itg
    if (itg.le.0)    itgex_hth(itg) = iabs(itg) + 1
    if (itg.ge.ntg+1) itgex_hth(itg) = 2*ntg - itg + 1
  end do
  do itg = -2, ntg + 2 
    do ipg = 1, npg
      if (itg.ge.1.and.itg.le.ntg)  ipgex_hth(ipg,itg) = ipg
      if (itg.le.0.or.itg.ge.ntg+1) ipgex_hth(ipg,itg) = mod(ipg + npg/2,npg)
    end do
  end do
!
  do ipg = -2, npg + 2
    if (ipg.ge.1.and.ipg.le.npg) ipgex_hphi(ipg) = ipg
    if (ipg.le.0)     ipgex_hphi(ipg) = npg + ipg
    if (ipg.ge.npg+1) ipgex_hphi(ipg) = ipg - npg
  end do
!
end subroutine grid_extended
end module coordinate_grav_extended
