module interface_IO_input_grav_export_Faraday
  implicit none
  interface 
    subroutine IO_input_grav_export_Faraday(filenm,fxd,fyd,fzd,fxyd,fxzd,fyzd)
      real(8), pointer ::  fxd(:,:,:),  fyd(:,:,:),  fzd(:,:,:), &
          &               fxyd(:,:,:), fxzd(:,:,:), fyzd(:,:,:)
      character(len=*) :: filenm
    end subroutine IO_input_grav_export_Faraday
  end interface
end module interface_IO_input_grav_export_Faraday
