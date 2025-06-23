subroutine interpo_gr2fl_metric_CF_export(alph,psi,bvxd,bvyd,bvzd,alphf,psif,bvxdf,bvydf,bvzdf,rs)
!  use def_metric, only : alph, psi, bvxd, bvyd, bvzd
!  use def_metric_on_SFC_CF
  use interface_interpo_gr2fl_export
  implicit none
  real(8), pointer :: psi(:,:,:), alph(:,:,:), bvxd(:,:,:), bvyd(:,:,:), bvzd(:,:,:)
  real(8), pointer :: psif(:,:,:), alphf(:,:,:), bvxdf(:,:,:), bvydf(:,:,:), bvzdf(:,:,:), rs(:,:)
 !
  call interpo_gr2fl_export(alph, alphf, rs)
  call interpo_gr2fl_export(psi , psif , rs)
  call interpo_gr2fl_export(bvxd, bvxdf, rs)
  call interpo_gr2fl_export(bvyd, bvydf, rs)
  call interpo_gr2fl_export(bvzd, bvzdf ,rs)
!
!  write(6,*) "gr2fl:", psif(1,1,1), alphf(1,1,1), bvxdf(1,1,1)

end subroutine interpo_gr2fl_metric_CF_export
