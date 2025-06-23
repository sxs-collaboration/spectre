subroutine IO_input_star4ve_export(filenm,utf,uxf,uyf,uzf)
  use phys_constant, only : long
  implicit none
  integer :: ir, it, ip, nrtmp, nttmp, nptmp
  real(8), pointer :: utf(:,:,:), uxf(:,:,:), uyf(:,:,:), uzf(:,:,:)
  character(len=*) :: filenm
!
  write(6,*) "Reading fluid 4-velocity..."
! ---.
  open(13,file=trim(filenm),status='old')
  read(13,'(5i5)') nrtmp, nttmp, nptmp
  do ip = 0, nptmp
    do it = 0, nttmp
      do ir = 0, nrtmp
        read(13,'(1p,6e20.12)')  utf(ir,it,ip), &
        &                        uxf(ir,it,ip), &
        &                        uyf(ir,it,ip), &
        &                        uzf(ir,it,ip)
      end do
    end do
  end do
  close(13)
!
end subroutine IO_input_star4ve_export
