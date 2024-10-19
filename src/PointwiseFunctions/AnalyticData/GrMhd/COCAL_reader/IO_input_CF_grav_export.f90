subroutine IO_input_CF_grav_export(filenm,psi,alph,bvxd,bvyd,bvzd)
  use phys_constant, only : long, nnrg, nntg, nnpg
  implicit none
  integer :: ir, it, ip, nrtmp, nttmp, nptmp
  real(8), pointer :: psi(:,:,:), alph(:,:,:), bvxd(:,:,:), bvyd(:,:,:), bvzd(:,:,:)
  character(len=*) :: filenm
!
  write(6,*) "Reading psi, alpha, beta_i..."
! --- Metric potentials.
  open(13,file=trim(filenm),status='old')
  read(13,'(5i5)') nrtmp, nttmp, nptmp
  do ip = 0, nptmp
    do it = 0, nttmp
      do ir = 0, nrtmp
        read(13,'(1p,6e20.12)')  psi(ir,it,ip), &
    &                           alph(ir,it,ip), &
    &                           bvxd(ir,it,ip), &
    &                           bvyd(ir,it,ip), &
    &                           bvzd(ir,it,ip)
      end do
    end do
  end do
  close(13)
!
end subroutine IO_input_CF_grav_export
