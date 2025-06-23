module interface_IO_input_gradvep_export 
  implicit none
  interface
    subroutine IO_input_gradvep_export(filenm, vepxf, vepyf, vepzf)
      real(8), pointer :: vepxf(:,:,:), vepyf(:,:,:), vepzf(:,:,:)
      character(len=*) :: filenm
    end subroutine IO_input_gradvep_export
  end interface
end module interface_IO_input_gradvep_export
