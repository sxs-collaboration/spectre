subroutine interpo_gr2cgr_4th(fnc,cfn,xc,yc,zc)
  use phys_constant, only : long, pi
  use grid_parameter, only : nrg, ntg, npg
  use coordinate_grav_extended
  implicit none
  real(long), pointer     :: fnc(:,:,:)
  real(long), intent(out) :: cfn
  real(long) ::  xc, yc, zc, rc, thc, phic, varpic
  real(long) ::  r4(4), th4(4), phi4(4), fr4(4), ft4(4), fp4(4)
  integer :: irg, itg, ipg, irgex, itgex, ipgex
  integer :: ir0, it0, ip0, irg0 , itg0 , ipg0, ii, jj, kk
  real(long), external :: lagint_4th
!
! --- Interpolation from the central spherical grid 
! --- to a Cartesian grid point using 4th order Lagrange formula.
!
  cfn = 0.0d0
!
  rc     = dsqrt(dabs(xc**2 + yc**2 + zc**2))
  varpic = dsqrt(dabs(xc**2 + yc**2))
  thc  = dmod(2.0d0*pi + datan2(varpic,zc),2.0d0*pi)
  phic = dmod(2.0d0*pi + datan2(    yc,xc),2.0d0*pi)
!
  do irg = 0, nrg+1
    if (rc.lt.rgex(irg).and.rc.ge.rgex(irg-1)) ir0 = min0(irg-2,nrg-3)
  end do
  do itg = 0, ntg+1
    if (thc.lt.thgex(itg).and.thc.ge.thgex(itg-1)) it0 = itg-2
  end do
  do ipg = 0, npg+1
    if (phic.lt.phigex(ipg).and.phic.ge.phigex(ipg-1)) ip0 = ipg-2
  end do
!
  do ii = 1, 4
    irg0 = ir0 + ii - 1
    itg0 = it0 + ii - 1
    ipg0 = ip0 + ii - 1
    r4(ii) = rgex(irg0)
    th4(ii) = thgex(itg0)
    phi4(ii) = phigex(ipg0)
  end do
!
! --  Center
!  if (rc.eq.0.0d0) then
!    cfn = fnc(0,0,0)
!    return
!  end if
!
! --  general
!
  do kk = 1, 4
    ipg0 = ip0 + kk - 1
    do jj = 1, 4
      itg0 = it0 + jj - 1
      do ii = 1, 4
        irg0 = ir0 + ii - 1
        irgex = irgex_r(irg0)
        itgex = itgex_r(itgex_th(itg0),irg0)
        ipgex = ipgex_r(ipgex_th(ipgex_phi(ipg0),itg0),irg0)
        fr4(ii) = fnc(irgex,itgex,ipgex)
      end do
      ft4(jj) = lagint_4th(r4,fr4,rc)
    end do
    fp4(kk) = lagint_4th(th4,ft4,thc)
  end do
  cfn = lagint_4th(phi4,fp4,phic)
!
end subroutine interpo_gr2cgr_4th
