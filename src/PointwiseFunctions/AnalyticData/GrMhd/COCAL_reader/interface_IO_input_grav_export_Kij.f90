module interface_IO_input_grav_export_Kij
  implicit none
  interface 
    subroutine IO_input_grav_export_Kij(filenm,kxx,kxy,kxz,kyy,kyz,kzz)
      real(8), pointer :: kxx(:,:,:), kxy(:,:,:), kxz(:,:,:), &
          &               kyy(:,:,:), kyz(:,:,:), kzz(:,:,:)
      character(len=*) :: filenm
    end subroutine IO_input_grav_export_Kij
  end interface
end module interface_IO_input_grav_export_Kij
