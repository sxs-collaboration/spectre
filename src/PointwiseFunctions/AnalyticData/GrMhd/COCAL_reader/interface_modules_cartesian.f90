module interface_modules_cartesian
  use phys_constant, only : long
  implicit none
  interface 
!______________________________________________________________________
    subroutine interpo_fl2cgr_4th(fnc,cfn,xc,yc,zc)
      real(8), pointer     :: fnc(:,:,:)
      real(8), intent(out) :: cfn
      real(8) ::  xc, yc, zc
    end subroutine interpo_fl2cgr_4th

    subroutine interpo_fl2cgr_4th_export(fnc,cfn,xc,yc,zc,rs)
      real(8), pointer     :: fnc(:,:,:), rs(:,:)
      real(8), intent(out) :: cfn
      real(8) ::  xc, yc, zc
    end subroutine interpo_fl2cgr_4th_export

    subroutine interpo_gr2cgr_4th(fnc,cfn,xc,yc,zc)
      real(8), pointer     :: fnc(:,:,:)
      real(8), intent(out) :: cfn
      real(8) ::  xc, yc, zc
    end subroutine interpo_gr2cgr_4th
    subroutine interpolation_matter(fnc,fncca)
      real(8), pointer :: fnc(:,:,:)
      real(8), pointer :: fncca(:,:,:)
    end subroutine interpolation_matter
    subroutine interpolation_metric(fnc,fncca)
      real(8), pointer :: fnc(:,:,:)
      real(8), pointer :: fncca(:,:,:)
    end subroutine interpolation_metric
    subroutine interpolation_metric_bh(fnc,fncca)
      real(8), pointer :: fnc(:,:,:)
      real(8), pointer :: fncca(:,:,:)
    end subroutine interpolation_metric_bh
    subroutine interpolation_fillup_cartesian(fnc,fncca)
      real(8), pointer :: fnc(:,:,:)
      real(8), pointer :: fncca(:,:,:)
    end subroutine interpolation_fillup_cartesian
    subroutine interpolation_fillup_cartesian_bh(fnc,fncca)
      real(8), pointer :: fnc(:,:,:)
      real(8), pointer :: fncca(:,:,:)
    end subroutine interpolation_fillup_cartesian_bh
    subroutine interpolation_fillup_cartesian_bh_parity(fnc,fncca,par)
      real(8), pointer :: fnc(:,:,:)
      real(8), pointer :: fncca(:,:,:)
      real(8) :: par
    end subroutine interpolation_fillup_cartesian_bh_parity
    subroutine interpolation_fillup_cartesian_parity(fnc,fncca,par)
      real(8), pointer :: fnc(:,:,:)
      real(8), pointer :: fncca(:,:,:)
      real(8) :: par
    end subroutine interpolation_fillup_cartesian_parity
    subroutine interpolation_fillup_cartesian_mpt(fnc,fncca,impt,impt_ex)
      real(8), pointer :: fnc(:,:,:)
      real(8), pointer :: fncca(:,:,:)
      integer :: impt, impt_ex
    end subroutine interpolation_fillup_cartesian_mpt
    subroutine interpolation_fillup_cartesian_parity_BNS_mpt(fnc,fncca,impt,impt_ex,par)
      real(8), pointer :: fnc(:,:,:)
      real(8), pointer :: fncca(:,:,:)
      integer :: impt, impt_ex
      real(8) :: par
    end subroutine interpolation_fillup_cartesian_parity_BNS_mpt
!______________________________________________________________________
  end interface
end module interface_modules_cartesian
