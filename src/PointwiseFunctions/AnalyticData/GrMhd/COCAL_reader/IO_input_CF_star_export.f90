subroutine IO_input_CF_star_export(filenm,emd,rs,omef,ome,ber,radi)
  use phys_constant, only : long, nnrg, nntg, nnpg
  implicit none
  integer :: ir, it, ip, nrtmp, nttmp, nptmp
  real(8), pointer :: emd(:,:,:), rs(:,:), omef(:,:,:)
  real(8) :: ome, ber, radi
  character(len=*) :: filenm
!
! --- Matter
    write(6,*) "Reading emd, rs, omef..."
    open(12,file=trim(filenm),status='old')
    read(12,'(5i5)')  nrtmp, nttmp, nptmp
    do ip = 0, nptmp
      do it = 0, nttmp
        do ir = 0, nrtmp
          read(12,'(1p,6e20.12)') emd(ir,it,ip), rs(it,ip), omef(ir,it,ip)
        end do
      end do
    end do
    read(12,'(1p,6e20.12)') ome, ber, radi
    close(12)
!
end subroutine IO_input_CF_star_export
