module def_peos_parameter
  use phys_constant         !nnpeos
  implicit none
  real(8) :: abc(0:nnpeos), abi(0:nnpeos), rhoi(0:nnpeos), &
  &          qi(0:nnpeos), hi(0:nnpeos)
  real(8) :: rhocgs(0:nnpeos), abccgs(0:nnpeos)
  real(8) :: rhoini_cgs, rhoini_gcm1, emdini_gcm1  !used in TOV solver
  real(8) :: sgma, constqc, cbar                   !used in quark core
  integer :: nphase
end module def_peos_parameter
