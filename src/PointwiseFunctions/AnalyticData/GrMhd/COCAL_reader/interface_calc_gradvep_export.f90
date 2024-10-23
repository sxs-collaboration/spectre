module interface_calc_gradvep_export
  implicit none
  interface 
    subroutine calc_gradvep_export(potf,potxf,potyf,potzf,rs)
      real(8), pointer ::  potf(:,:,:), potxf(:,:,:), potyf(:,:,:), potzf(:,:,:), rs(:,:)
    end subroutine calc_gradvep_export
  end interface
end module interface_calc_gradvep_export
