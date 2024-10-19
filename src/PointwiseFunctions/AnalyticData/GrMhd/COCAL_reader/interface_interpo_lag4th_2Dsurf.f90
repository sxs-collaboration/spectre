module interface_interpo_lag4th_2Dsurf
  implicit none
  interface 
    subroutine interpo_lag4th_2Dsurf(val,fnc,tv,pv)
      real(8), intent(out) :: val
      real(8), intent(in)  :: tv, pv
      real(8), pointer :: fnc(:,:)
    end subroutine interpo_lag4th_2Dsurf
  end interface
end module interface_interpo_lag4th_2Dsurf
