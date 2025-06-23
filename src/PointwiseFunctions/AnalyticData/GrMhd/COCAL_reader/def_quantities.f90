module def_quantities
  use phys_constant, only : long
  implicit none
  real(long) :: admmass, komarmass, komarmass_nc, restmass, propermass, angmom
  real(long) :: admmass_asymp, komarmass_asymp, angmom_asymp, admmom_asymp(3)
  real(long) :: angmom_asympto(3), anmo(3)
  real(long) :: charge, charge_asymp
!
  real(long) :: T_kinene, W_gravene, P_intene, M_emfene
  real(long) :: M_torBene, M_polBene, M_eleEene, Virial
  real(long) :: ToverW, PoverW, MoverW
  real(long) :: T_kinene_omeJ, W_gravene_omeJ, ToverW_omeJ
  real(long) :: MtorBoverW, MpolBoverW, MeleEoverW, I_inertia
  real(long) :: gravmass_sph, restmass_sph, propermass_sph
  real(long) :: MoverR_sph, schwarz_radi_sph, schwarz_radi_sph_km
!
  real(long) :: admmass_thr, angmom_thr, angmom_smarr, komarmass_thr
  real(long) :: angmom_throat(3)
  real(long) :: app_hor_area_bh, irredmass, christmass, bindingene
  real(long) :: qua_loc_spin, qua_loc_spin_surf
  real(long) :: circ_shift_xy
  real(long) :: circ_line_xy, circ_line_yz, circ_line_zx
  real(long) :: circ_surf_xy, circ_surf_yz, circ_surf_zx
!
  real(long) :: coord_radius_x,  coord_radius_y,  coord_radius_z
  real(long) :: proper_radius_x, proper_radius_y, proper_radius_z
  real(long) :: rho_c, pre_c, epsi_c, q_c
  real(long) :: rho_max, pre_max, epsi_max, q_max
!
  real(long) :: coord_radius_x_km,  coord_radius_y_km,  coord_radius_z_km
  real(long) :: proper_radius_x_km, proper_radius_y_km, proper_radius_z_km
  real(long) :: rho_c_cgs, pre_c_cgs, epsi_c_cgs, q_c_cgs
  real(long) :: rho_max_cgs, pre_max_cgs, epsi_max_cgs, q_max_cgs
  real(long) :: ome_cgs
! -- red & blue shifts 
  real(long) :: zrb_xp_plus, zrb_xp_minus  ! at the surface on x axis
  real(long) :: zrb_yp_plus, zrb_yp_minus  ! at the surface on y axis
  real(long) :: zrb_zp_plus, zrb_zp_minus  ! at the surface on z axis
! -- enthalpy at the surface of x,y,z-axis
  real(long) :: dhdr_x, dhdr_y, dhdr_z  ! at the surface
  real(long) :: chi_cusp
!
  real(long) :: Iij(1:3,1:3), Itf(1:3,1:3)
  real(long) :: dt1Itf(1:3,1:3), dt2Itf(1:3,1:3), dt3Itf(1:3,1:3)
  real(long) :: LGW, dJdt(1:3), hplus, hcross
!
end module def_quantities
