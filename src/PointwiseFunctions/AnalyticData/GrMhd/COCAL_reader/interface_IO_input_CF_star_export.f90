module interface_IO_input_CF_star_export
  implicit none
  interface 
    subroutine IO_input_CF_star_export(filenm,emd,rs,omef,ome,ber,radi)
      real(8), pointer :: emd(:,:,:), rs(:,:), omef(:,:,:)
      real(8) ::  ome,ber,radi
      character(len=*) :: filenm
    end subroutine IO_input_CF_star_export
  end interface
end module interface_IO_input_CF_star_export
