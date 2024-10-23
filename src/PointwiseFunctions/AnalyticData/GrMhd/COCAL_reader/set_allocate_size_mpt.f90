subroutine set_allocate_size_mpt
  use phys_constant, only : nmpt
  use grid_parameter, only : nrg, ntg, npg, nlg,&
                           & nrf, ntf, npf, nlf
!
  implicit none
  integer :: nrgmax=0, ntgmax=0, npgmax=0, nlgmax=0,&
           & nrfmax=0, ntfmax=0, npfmax=0, nlfmax=0
  integer :: impt
!
  do impt = 1, nmpt
    call copy_grid_parameter_from_mpt(impt)
!
    nrgmax = max(nrgmax, nrg)
    ntgmax = max(ntgmax, ntg)
    npgmax = max(npgmax, npg)
    nlgmax = max(nlgmax, nlg)
    nrfmax = max(nrfmax, nrf)
    ntfmax = max(ntfmax, ntf)
    npfmax = max(npfmax, npf)
    nlfmax = max(nlfmax, nlf)
  end do
!
  nrg =nrgmax
  ntg =ntgmax
  npg =npgmax
  nlg =nlgmax
  nrf =nrfmax
  ntf =ntfmax
  npf =npfmax
  nlf =nlfmax
!
end subroutine set_allocate_size_mpt
