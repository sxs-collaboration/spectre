subroutine IO_input_grav_export_Ai(filenm,va,vaxd,vayd,vazd)
  use phys_constant, only : long
  implicit none
  integer :: ir, it, ip, nrtmp, nttmp, nptmp
  real(8), pointer :: va(:,:,:), vaxd(:,:,:), vayd(:,:,:), vazd(:,:,:)
  character(len=*) :: filenm
!
  write(6,*) "Reading va, vaxd, vayd, vazd..."
! --- EMF potentials.
  open(13,file=trim(filenm),status='old')
  read(13,'(5i5)') nrtmp, nttmp, nptmp
  do ip = 0, nptmp
    do it = 0, nttmp
      do ir = 0, nrtmp
        read(13,'(1p,6e20.12)')    va(ir,it,ip), &
        &                        vaxd(ir,it,ip), &
        &                        vayd(ir,it,ip), &
        &                        vazd(ir,it,ip)
      end do
    end do
  end do
  close(13)
!
end subroutine IO_input_grav_export_Ai
