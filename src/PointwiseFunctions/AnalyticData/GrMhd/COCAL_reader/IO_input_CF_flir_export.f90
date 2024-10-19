subroutine IO_input_CF_flir_export(filenm,emd,vep,ome,ber,radi)
  use phys_constant, only : long, nnrg, nntg, nnpg
  implicit none
  integer :: ir, it, ip, nrtmp, nttmp, nptmp
  real(8), pointer :: emd(:,:,:), vep(:,:,:), rs(:,:)
  real(8) :: ome, ber, radi
  character(len=*) :: filenm
  character*400 :: tmp_file
!
! --- Matter
!  tmp_file = "/home/fs01/ml2847/SPECTRE/BNS/ID/IRE3.0_SLy_SLy_010_M1.96/work_area_BNS/bnsflu_3D_mpt1.las"
  open(12,file=trim(filenm),status='old')
!  open(12,file=trim(tmp_file),status='old')
  read(12,'(5i5)')  nrtmp, nttmp, nptmp
  do ip = 0, nptmp
    do it = 0, nttmp
      do ir = 0, nrtmp
        read(12,'(1p,6e20.12)') emd(ir,it,ip), vep(ir,it,ip)
      end do
    end do
  end do
  read(12,'(1p,6e20.12)') ome, ber, radi
  close(12)
!
end subroutine IO_input_CF_flir_export
