module def_bht_parameter
  use phys_constant, only : long
  implicit none
  real(long) ::  mass_bh_crist, emdc_ratio
  real(long) ::  Mt_over_Mbh, angmom_torus, angmom_bh(3), am_bh
  real(long) ::  disk_height, disk_xh, disk_yh, disk_zh
  real(long) ::  disk_width, disk_xin, disk_yin, disk_zin, &
     &                       disk_xou, disk_you, disk_zou
  real(long) ::  xe_c, ye_c, ze_c, xe_m, ye_m, ze_m, &
     &           o_c, j_c, l_c, o_m, j_m, l_m, &
     &           o_in,j_in,l_in, o_out,j_out,l_out 
end module def_bht_parameter
