subroutine IO_input_CF_surf_export(filenm,rs)
  use phys_constant, only : long, nnrg, nntg, nnpg
  implicit none
  integer :: ir, it, ip, nrtmp, nttmp, nptmp
  real(8), pointer ::  rs(:,:)
  character(len=*) :: filenm
  character*400 ::tmp_file
!
! --- Star surface
 ! tmp_file="/home/fs01/ml2847/SPECTRE/BNS/ID/IRE3.0_SLy_SLy_010_M1.96/work_area_BNS/bnssur_3D_mpt1.las"
  open(15,file=trim(filenm),status='old')
!  open(15,file=trim(tmp_file),status='old')

  read(15,'(5i5)')   nttmp,  nptmp
  do ip = 0, nptmp
    do it = 0, nttmp
      read(15,'(1p,6e20.12)')   rs(it,ip)
    end do
  end do
  close(15)
!
end subroutine IO_input_CF_surf_export
