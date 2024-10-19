! extended coordinate for the field
!______________________________________________
subroutine copy_coordinate_grav_extended_from_mpt(impt)
  use phys_constant, only : long, nnrg, nntg, nnpg
!  use grid_parameter, only : nrg, ntg, npg
  use coordinate_grav_extended
  use coordinate_grav_extended_mpt
  use copy_array_static_2dto1d_mpt
  use copy_int_array_static_2dto1d_mpt
  use copy_int_array_static_3dto2d_mpt
  implicit none
  integer :: impt
!
  call copy_arraystatic_2dto1d_mpt(impt,rgex_,rgex,-2,nnrg+2)
  call copy_arraystatic_2dto1d_mpt(impt,thgex_,thgex,-2,nntg+2)
  call copy_arraystatic_2dto1d_mpt(impt,phigex_,phigex,-2,nnpg+2)
  call copy_int_arraystatic_2dto1d_mpt(impt,irgex_r_,irgex_r,-2,nnrg+2)
  call copy_int_arraystatic_3dto2d_mpt(impt,itgex_r_,itgex_r, 0,nntg,-2,nnrg+2)
  call copy_int_arraystatic_3dto2d_mpt(impt,ipgex_r_,ipgex_r, 0,nnpg,-2,nnrg+2)
  call copy_int_arraystatic_2dto1d_mpt(impt,itgex_th_,itgex_th,-2,nntg+2)
  call copy_int_arraystatic_3dto2d_mpt(impt,ipgex_th_,ipgex_th, 0,nnpg,-2,nntg+2)
  call copy_int_arraystatic_2dto1d_mpt(impt,ipgex_phi_,ipgex_phi,-2,nnpg+2)
!
  call copy_arraystatic_2dto1d_mpt(impt,hrgex_,hrgex,-2,nnrg+2)
  call copy_arraystatic_2dto1d_mpt(impt,hthgex_,hthgex,-2,nntg+2)
  call copy_arraystatic_2dto1d_mpt(impt,hphigex_,hphigex,-2,nnpg+2)
  call copy_int_arraystatic_2dto1d_mpt(impt,irgex_hr_,irgex_hr,-2,nnrg+2)
  call copy_int_arraystatic_3dto2d_mpt(impt,itgex_hr_,itgex_hr, 1,nntg,-2,nnrg+2)
  call copy_int_arraystatic_3dto2d_mpt(impt,ipgex_hr_,ipgex_hr, 1,nnpg,-2,nnrg+2)
  call copy_int_arraystatic_2dto1d_mpt(impt,itgex_hth_,itgex_hth,-2,nntg+2)
  call copy_int_arraystatic_3dto2d_mpt(impt,ipgex_hth_,ipgex_hth,1,nnpg,-2,nntg+2)
  call copy_int_arraystatic_2dto1d_mpt(impt,ipgex_hphi_,ipgex_hphi,-2,nnpg+2)
!
end subroutine copy_coordinate_grav_extended_from_mpt
