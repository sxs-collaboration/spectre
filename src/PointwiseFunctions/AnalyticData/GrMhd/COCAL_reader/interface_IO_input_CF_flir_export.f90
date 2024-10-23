module interface_IO_input_CF_flir_export
  implicit none
  interface 
    subroutine IO_input_CF_flir_export(filenm, emd,vep,ome,ber,radi)
      real(8), pointer :: emd(:,:,:), vep(:,:,:)
      real(8) ::  ome,ber,radi
      character(len=*) :: filenm
    end subroutine IO_input_CF_flir_export
  end interface
end module interface_IO_input_CF_flir_export
