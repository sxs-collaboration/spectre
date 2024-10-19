module interface_IO_input_star4ve_export
  implicit none
  interface 
    subroutine IO_input_star4ve_export(filenm, utf,uxf,uyf,uzf)
      real(8), pointer :: utf(:,:,:), uxf(:,:,:), uyf(:,:,:), uzf(:,:,:)
      character(len=*) :: filenm
    end subroutine IO_input_star4ve_export
  end interface
end module interface_IO_input_star4ve_export
