module interface_IO_input_grav_export_Ai
  implicit none
  interface 
    subroutine IO_input_grav_export_Ai(filenm, va,vaxd,vayd,vazd)
      real(8), pointer :: va(:,:,:), vaxd(:,:,:), vayd(:,:,:), vazd(:,:,:)
      character(len=*) :: filenm
    end subroutine IO_input_grav_export_Ai
  end interface
end module interface_IO_input_grav_export_Ai
