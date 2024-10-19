subroutine peos_lookup(qp,qpar,iphase)
!
  use phys_constant			!nnpeos
  use def_peos_parameter	!abc,abi,rhoi,qi,hi,nphase
  implicit none
!
  real(8), intent(in)  :: qp, qpar(0:nnpeos)
  real(8)              :: det
  integer, intent(out) :: iphase
  integer              :: ii
!
! --  Monotonically increasing qpar is assumed.
!
  iphase = 1
  do ii = 1, nphase
    det = (qp-qpar(ii))*(qp-qpar(ii-1))
    if (det <= 0.0d0) then
      iphase = ii
      exit
    end if
  end do
!
end subroutine peos_lookup
