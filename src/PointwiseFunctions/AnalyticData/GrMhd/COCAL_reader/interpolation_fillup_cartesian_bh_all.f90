subroutine interpolation_fillup_cartesian_bh_all
  use phys_constant, only : long
  use def_metric
  use def_metric_cartesian  
  use grid_parameter_cartesian, only : nx, ny, nz
  use grid_parameter_binary_excision, only : ex_rgmid, ex_radius 
  use coordinate_grav_xyz, only : x, y, z
  use interface_modules_cartesian
  use grid_parameter, only : rgin
  implicit none
  real(long) :: xc, yc, zc, cfn, R
  integer :: ix, iy, iz
  real(long) :: r1,r2, cpsi,calph,cbvxd,cbvyd,cbvzd
!
  do iz = 1, nz
    zc = z(iz)
    do iy = 1, ny
      yc = y(iy)
      do ix = 1, nx
        xc = x(ix)
        r1 = sqrt(xc**2 + yc**2 + zc**2)
        r2 = sqrt((xc-ex_rgmid)**2 + yc**2 + zc**2)
        if((r1.ge.rgin).and.(r2.ge.rgin)) then
          call interpo_gr2cgr_4th(psi,cpsi,xc,yc,zc)
          call interpo_gr2cgr_4th(alph,calph,xc,yc,zc)
          call interpo_gr2cgr_4th(bvxd,cbvxd,xc,yc,zc)
          call interpo_gr2cgr_4th(bvyd,cbvyd,xc,yc,zc)
          call interpo_gr2cgr_4th(bvzd,cbvzd,xc,yc,zc)
          psica(ix,iy,iz)  = cpsi
          alphca(ix,iy,iz) = calph
          bvxdca(ix,iy,iz) = cbvxd
          bvydca(ix,iy,iz) = cbvyd
          bvzdca(ix,iy,iz) = cbvzd
        else
          psica(ix,iy,iz)  = 0.0d0
          alphca(ix,iy,iz) = 0.0d0
          bvxdca(ix,iy,iz) = 0.0d0
          bvydca(ix,iy,iz) = 0.0d0
          bvzdca(ix,iy,iz) = 0.0d0
        endif
      end do
    end do
  end do
!
  do iz = 1, nz
    zc = z(iz)
    do iy = 1, ny
      yc = y(iy)
      do ix = 1, nx
        xc = x(ix)
        R = sqrt((xc-ex_rgmid)**2 + yc**2 + zc**2)
        if ((R <= ex_radius*1.2d0).and.(R>=rgin)) then
          call interpo_gr2cgr_4th(psi ,cpsi ,-xc+ex_rgmid,-yc,zc)
          call interpo_gr2cgr_4th(alph,calph,-xc+ex_rgmid,-yc,zc)
          call interpo_gr2cgr_4th(bvxd,cbvxd,-xc+ex_rgmid,-yc,zc)
          call interpo_gr2cgr_4th(bvyd,cbvyd,-xc+ex_rgmid,-yc,zc)
          call interpo_gr2cgr_4th(bvzd,cbvzd,-xc+ex_rgmid,-yc,zc)
          psica(ix,iy,iz)  = cpsi
          alphca(ix,iy,iz) = calph
          bvxdca(ix,iy,iz) = -cbvxd
          bvydca(ix,iy,iz) = -cbvyd
          bvzdca(ix,iy,iz) = +cbvzd
        endif
      end do
    end do
  end do
end subroutine interpolation_fillup_cartesian_bh_all
