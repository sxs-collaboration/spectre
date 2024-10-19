module interface_IO_input_CF_flco_export
  implicit none
  interface 
    subroutine IO_input_CF_flco_export(filenm, emd,ome,ber,radi)
      real(8), pointer :: emd(:,:,:)
      real(8) ::  ome,ber,radi
      character(len=*) :: filenm
    end subroutine IO_input_CF_flco_export
  end interface
end module interface_IO_input_CF_flco_export
