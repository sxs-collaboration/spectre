subroutine IO_input_grav_export_Kij(filenm,kxx,kxy,kxz,kyy,kyz,kzz)
  use phys_constant, only : long
  implicit none
  integer :: irg, itg, ipg, nrtmp, nttmp, nptmp
  real(8), pointer :: kxx(:,:,:), kxy(:,:,:), kxz(:,:,:), &
      &               kyy(:,:,:), kyz(:,:,:), kzz(:,:,:)
  character(len=*) :: filenm
  character*400 ::tmp_file
!
  write(6,*) "Reading Kij..."
! --- Metric potentials.
!  tmp_file="/home/fs01/ml2847/SPECTRE/BNS/ID/IRE3.0_SLy_SLy_010_M1.96/work_area_BNS/bnsgra_Kij_3D_mpt1.las"
  open(13,file=trim(filenm),status='old')
!  open(13,file=trim(tmp_file),status='old')
  read(13,'(5i5)')  nrtmp, nttmp, nptmp
  do ipg = 0, nptmp
    do itg = 0, nttmp
      do irg = 0, nrtmp
        read(13,'(1p,6e23.15)')  kxx(irg,itg,ipg), &
        &                        kxy(irg,itg,ipg), &
        &                        kxz(irg,itg,ipg), &
        &                        kyy(irg,itg,ipg), &
        &                        kyz(irg,itg,ipg), &
        &                        kzz(irg,itg,ipg)
      end do
    end do
  end do
  close(13)
!
end subroutine IO_input_grav_export_Kij
