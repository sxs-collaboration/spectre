module interface_IO_input_CF_surf_export
  implicit none
  interface 
    subroutine IO_input_CF_surf_export(filenm, rs)
      real(8), pointer :: rs(:,:)
      character(len=*) :: filenm
    end subroutine IO_input_CF_surf_export
  end interface
end module interface_IO_input_CF_surf_export
