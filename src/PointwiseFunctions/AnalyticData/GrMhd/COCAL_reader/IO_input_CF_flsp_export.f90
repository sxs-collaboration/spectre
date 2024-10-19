subroutine IO_input_CF_flsp_export(filenm,emd,vep,wxspf,wyspf,wzspf,ome,ber,radi,confpow)
  use phys_constant, only : long, nnrg, nntg, nnpg
  implicit none
  integer :: ir, it, ip, nrtmp, nttmp, nptmp
  real(8), pointer :: emd(:,:,:), vep(:,:,:), wxspf(:,:,:), wyspf(:,:,:), wzspf(:,:,:)
  real(8) :: ome, ber, radi, confpow, omespx, omespy, omespz
  character(len=*) :: filenm
!
! --- Matter
  open(12,file=trim(filenm),status='old')
  read(12,'(5i5)')  nrtmp, nttmp, nptmp
  do ip = 0, nptmp
    do it = 0, nttmp
      do ir = 0, nrtmp
          read(12,'(1p,6e20.12)') emd(ir,it,ip), vep(ir,it,ip), wxspf(ir,it,ip), &
                &   wyspf(ir,it,ip), wzspf(ir,it,ip)
      end do
    end do
  end do
  read(12,'(1p,6e20.12)') ome, ber, radi
  read(12,'(1p,6e20.12)') confpow, omespx, omespy, omespz
  close(12)
!
end subroutine IO_input_CF_flsp_export
