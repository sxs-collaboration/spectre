subroutine interpolation_fillup_cartesian_mpt(fnc, fncca, impt, impt_ex)
  use phys_constant, only : long
  use grid_parameter_cartesian, only : nx, ny, nz
  use grid_parameter_binary_excision, only : ex_rgmid, ex_radius 
  use coordinate_grav_xyz, only : x, y, z
  use def_binary_parameter, only : dis
  use interface_modules_cartesian
  implicit none
  real(long), pointer :: fnc(:,:,:)
  real(long), pointer :: fncca(:,:,:)
  real(long) :: xc, yc, zc, cfn, R
  integer :: ix, iy, iz, impt, impt_ex
!
  do iz = 1, nz
    zc = z(iz)
    do iy = 1, ny
      yc = y(iy)
      do ix = 1, nx
        xc = x(ix)
        call interpo_gr2cgr_4th(fnc,cfn,xc,yc,zc)
        fncca(ix,iy,iz) = cfn
        end do
    end do
  end do
!
  call copy_from_mpatch_interpolation_utility(impt_ex)
  call copy_poisson_solver_test_from_mpt(impt_ex)
  do iz = 1, nz
    zc = z(iz)
    do iy = 1, ny
      yc = - y(iy)
      do ix = 1, nx
        xc = x(ix)
        R = sqrt((xc-ex_rgmid)**2+ yc**2 + zc**2)
        if (R <= ex_radius*1.45d0.and.xc >= dis) then
          call interpo_gr2cgr_4th(fnc,cfn,-xc+ex_rgmid,-yc,zc)
          fncca(ix,iy,iz) = cfn
        endif
      end do
    end do
  end do
  call copy_from_mpatch_interpolation_utility(impt)
  call copy_poisson_solver_test_from_mpt(impt)
end subroutine interpolation_fillup_cartesian_mpt
