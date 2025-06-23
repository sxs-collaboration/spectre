module interface_interpo_gr2fl_export
  implicit none
  interface 
    subroutine interpo_gr2fl_export(grv,flv,rs)
      real(8), pointer :: grv(:,:,:), flv(:,:,:), rs(:,:)
    end subroutine interpo_gr2fl_export
  end interface
end module interface_interpo_gr2fl_export
