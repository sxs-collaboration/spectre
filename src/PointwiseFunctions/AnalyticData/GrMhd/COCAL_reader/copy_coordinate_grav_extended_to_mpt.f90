! extended coordinate for the field
!______________________________________________
subroutine copy_coordinate_grav_extended_to_mpt(impt)
  use phys_constant, only : long, nnrg, nntg, nnpg
!  use grid_parameter, only : nrg, ntg, npg
  use coordinate_grav_extended
  use coordinate_grav_extended_mpt
  use copy_array_static_1dto2d_mpt
  use copy_int_array_static_1dto2d_mpt
  use copy_int_array_static_2dto3d_mpt
  implicit none
  integer :: impt
!
  call copy_arraystatic_1dto2d_mpt(impt,rgex,rgex_,-2,nnrg+2)
  call copy_arraystatic_1dto2d_mpt(impt,thgex,thgex_,-2,nntg+2)
  call copy_arraystatic_1dto2d_mpt(impt,phigex,phigex_,-2,nnpg+2)
  call copy_int_arraystatic_1dto2d_mpt(impt,irgex_r,irgex_r_,-2,nnrg+2)
  call copy_int_arraystatic_2dto3d_mpt(impt,itgex_r,itgex_r_, 0,nntg,-2,nnrg+2)
  call copy_int_arraystatic_2dto3d_mpt(impt,ipgex_r,ipgex_r_, 0,nnpg,-2,nnrg+2)
  call copy_int_arraystatic_1dto2d_mpt(impt,itgex_th,itgex_th_,-2,nntg+2)
  call copy_int_arraystatic_2dto3d_mpt(impt,ipgex_th,ipgex_th_, 0,nnpg,-2,nntg+2)
  call copy_int_arraystatic_1dto2d_mpt(impt,ipgex_phi,ipgex_phi_,-2,nnpg+2)
!
  call copy_arraystatic_1dto2d_mpt(impt,hrgex,hrgex_,-2,nnrg+2)
  call copy_arraystatic_1dto2d_mpt(impt,hthgex,hthgex_,-2,nntg+2)
  call copy_arraystatic_1dto2d_mpt(impt,hphigex,hphigex_,-2,nnpg+2)
  call copy_int_arraystatic_1dto2d_mpt(impt,irgex_hr,irgex_hr_,-2,nnrg+2)
  call copy_int_arraystatic_2dto3d_mpt(impt,itgex_hr,itgex_hr_, 1,nntg,-2,nnrg+2)
  call copy_int_arraystatic_2dto3d_mpt(impt,ipgex_hr,ipgex_hr_, 1,nnpg,-2,nnrg+2)
  call copy_int_arraystatic_1dto2d_mpt(impt,itgex_hth,itgex_hth_,-2,nntg+2)
  call copy_int_arraystatic_2dto3d_mpt(impt,ipgex_hth,ipgex_hth_,1,nnpg,-2,nntg+2)
  call copy_int_arraystatic_1dto2d_mpt(impt,ipgex_hphi,ipgex_hphi_,-2,nnpg+2)
!
end subroutine copy_coordinate_grav_extended_to_mpt
