module interface_IO_input_CF_grav_export
  implicit none
  interface 
    subroutine IO_input_CF_grav_export(filenm, psi,alph,bvxd,bvyd,bvzd)
      real(8), pointer :: psi(:,:,:), alph(:,:,:), bvxd(:,:,:), bvyd(:,:,:), bvzd(:,:,:)
      character(len=*) :: filenm
    end subroutine IO_input_CF_grav_export
  end interface
end module interface_IO_input_CF_grav_export
