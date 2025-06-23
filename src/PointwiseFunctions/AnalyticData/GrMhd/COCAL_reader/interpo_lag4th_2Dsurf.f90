! Cubic (4th order) Lagrange interpolation of the surface.
subroutine interpo_lag4th_2Dsurf(val,fnc,tv,pv)
  use phys_constant, only : long
  use grid_parameter, only : ntg, npg
  use coordinate_grav_extended
  implicit none
  real(long), intent(out) :: val
  real(long), intent(in)  :: tv, pv
  real(long), pointer :: fnc(:,:)
  real(long) ::  th4(4), phi4(4), ft4(4), fp4(4)
  integer :: itg, ipg, itgex, ipgex
  integer :: it0, ip0, itg0 , ipg0, ii, jj, kk
  real(long), external :: lagint_4th
!
  do itg = 0, ntg+1
    if (tv.lt.thgex(itg).and.tv.ge.thgex(itg-1)) it0 = itg-2
  end do
  do ipg = 0, npg+1
    if (pv.lt.phigex(ipg).and.pv.ge.phigex(ipg-1)) ip0 = ipg-2
  end do
!
  do ii = 1, 4
    itg0 = it0 + ii - 1
    ipg0 = ip0 + ii - 1
    th4(ii) = thgex(itg0)
    phi4(ii) = phigex(ipg0)
  end do
!
  do kk = 1, 4
    ipg0 = ip0 + kk - 1
    do jj = 1, 4
      itg0 = it0 + jj - 1
      itgex = itgex_th(itg0)
      ipgex = ipgex_th(ipgex_phi(ipg0),itg0)
      ft4(jj) = fnc(itgex,ipgex)
    end do
    fp4(kk) = lagint_4th(th4,ft4,tv)
  end do
  val = lagint_4th(phi4,fp4,pv)
!
end subroutine interpo_lag4th_2Dsurf
