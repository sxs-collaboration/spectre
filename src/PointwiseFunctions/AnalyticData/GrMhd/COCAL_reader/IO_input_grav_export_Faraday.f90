subroutine IO_input_grav_export_Faraday(filenm,fxd,fyd,fzd,fxyd,fxzd,fyzd)
  use phys_constant, only : long
  implicit none
  integer :: ir, it, ip, nrtmp, nttmp, nptmp
  real(8), pointer :: fxd(:,:,:), fyd(:,:,:), fzd(:,:,:), fxyd(:,:,:), fxzd(:,:,:), fyzd(:,:,:)
  character(len=*) :: filenm
!
  write(6,*) "Reading Faraday tensor..."
! --- Faraday potentials.
  open(13,file=trim(filenm),status='old')
  read(13,'(5i5)') nrtmp, nttmp, nptmp
  do ip = 0, nptmp
    do it = 0, nttmp
      do ir = 0, nrtmp
        read(13,'(1p,6e20.12)')   fxd(ir,it,ip), &
        &                         fyd(ir,it,ip), &
        &                         fzd(ir,it,ip), &
        &                        fxyd(ir,it,ip), &
        &                        fxzd(ir,it,ip), &
        &                        fyzd(ir,it,ip)
      end do
    end do
  end do
  close(13)
!
end subroutine IO_input_grav_export_Faraday
