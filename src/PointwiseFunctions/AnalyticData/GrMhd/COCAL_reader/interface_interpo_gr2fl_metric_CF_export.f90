module interface_interpo_gr2fl_metric_CF_export
  implicit none
  interface 
    subroutine interpo_gr2fl_metric_CF_export(alph,psi,bvxd,bvyd,bvzd,alphf,psif,bvxdf,bvydf,bvzdf,rs)
      real(8), pointer :: psi(:,:,:), alph(:,:,:), bvxd(:,:,:), bvyd(:,:,:), bvzd(:,:,:)
      real(8), pointer :: psif(:,:,:), alphf(:,:,:), bvxdf(:,:,:), bvydf(:,:,:), bvzdf(:,:,:), rs(:,:)
    end subroutine interpo_gr2fl_metric_CF_export
  end interface
end module interface_interpo_gr2fl_metric_CF_export
