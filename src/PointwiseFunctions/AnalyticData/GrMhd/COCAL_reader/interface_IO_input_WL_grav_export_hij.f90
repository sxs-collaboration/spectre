module interface_IO_input_WL_grav_export_hij
  implicit none
  interface 
    subroutine IO_input_WL_grav_export_hij(filenm,hxxd,hxyd,hxzd,hyyd,hyzd,hzzd)
      real(8), pointer :: hxxd(:,:,:), hxyd(:,:,:), hxzd(:,:,:), &
          &               hyyd(:,:,:), hyzd(:,:,:), hzzd(:,:,:)
      character(len=*) :: filenm
    end subroutine IO_input_WL_grav_export_hij
  end interface
end module interface_IO_input_WL_grav_export_hij
