subroutine IO_input_gradvep_export(filenm, vepxf, vepyf, vepzf)                                                                                      
  use phys_constant, only : long
  implicit none
  integer :: ir, it, ip, nrtmp, nttmp, nptmp
  real(8), pointer :: vepxf(:,:,:), vepyf(:,:,:), vepzf(:,:,:)
  character(len=*) :: filenm
  character*400 ::tmp_file

!
  write(6,*) "Reading vepxf, vepyf, vepzf..."
!  tmp_file="/home/fs01/ml2847/SPECTRE/BNS/ID/IRE3.0_SLy_SLy_010_M1.96/work_area_BNS/bnsdvep_3D_mpt1.las"
  open(13,file=trim(filenm),status='old')
!  open(13,file=trim(tmp_file),status='old')
  read(13,'(5i5)') nrtmp, nttmp, nptmp
  do ip = 0, nptmp
    do it = 0, nttmp
      do ir = 0, nrtmp
        read(13,'(1p,3e20.12)')  vepxf(ir,it,ip), vepyf(ir,it,ip), vepzf(ir,it,ip)
      end do
    end do
  end do
  close(13)
!
end subroutine IO_input_gradvep_export
