subroutine invhij_WL_export(hxxd,hxyd,hxzd,hyyd,hyzd,hzzd,hxxu,hxyu,hxzu,hyyu,hyzu,hzzu)
  use grid_parameter, only : nrg, ntg, npg
  implicit none
  real(8), pointer :: hxxd(:,:,:), hxyd(:,:,:), hxzd(:,:,:), &
      &               hyyd(:,:,:), hyzd(:,:,:), hzzd(:,:,:), &
      &               hxxu(:,:,:), hxyu(:,:,:), hxzu(:,:,:), &
      &               hyyu(:,:,:), hyzu(:,:,:), hzzu(:,:,:)
  real(8) :: hxx, hxy, hxz, hyx, hyy, hyz, hzx, hzy, hzz, &
  &          hod1, hod2, hod3, detgm, detgmi, &
  &          gmxxu, gmxyu, gmxzu, gmyxu, gmyyu, gmyzu, &
  &          gmzxu, gmzyu, gmzzu
  integer :: ipg, itg, irg
!
  do ipg = 0, npg
    do itg = 0, ntg
      do irg = 0, nrg
!
        hxx = hxxd(irg,itg,ipg)
        hxy = hxyd(irg,itg,ipg)
        hxz = hxzd(irg,itg,ipg)
        hyx = hxy
        hyy = hyyd(irg,itg,ipg)
        hyz = hyzd(irg,itg,ipg)
        hzx = hxz
        hzy = hyz
        hzz = hzzd(irg,itg,ipg)
!
        hod1 = hxx + hyy + hzz
        hod2 = hxx*hyy + hxx*hzz + hyy*hzz &
        &    - hxy*hyx - hxz*hzx - hyz*hzy
        hod3 = hxx*hyy*hzz + hxy*hyz*hzx + hxz*hyx*hzy &
        &    - hxx*hyz*hzy - hxy*hyx*hzz - hxz*hyy*hzx
        detgm  = 1.0d0 + hod1 + hod2 + hod3
        detgmi = 1.0d0/detgm
!
        hod1  = + hyy + hzz
        hod2  = + hyy*hzz - hyz*hzy
        gmxxu = (1.0d0 + hod1 + hod2)*detgmi
        hod1  = - hxy
        hod2  = + hxz*hzy - hxy*hzz
        gmxyu = (hod1 + hod2)*detgmi
        hod1  = - hxz
        hod2  = + hxy*hyz - hxz*hyy
        gmxzu = (hod1 + hod2)*detgmi
        hod1  = + hxx + hzz
        hod2  = + hxx*hzz - hxz*hzx
        gmyyu = (1.0d0 + hod1 + hod2)*detgmi
        hod1  = - hyz
        hod2  = + hxz*hyx - hxx*hyz
        gmyzu = (hod1 + hod2)*detgmi
        hod1  = + hxx + hyy
        hod2  = + hxx*hyy - hxy*hyx
        gmzzu = (1.0d0 + hod1 + hod2)*detgmi
        gmyxu = gmxyu
        gmzxu = gmxzu
        gmzyu = gmyzu
!
        hxxu(irg,itg,ipg) = gmxxu - 1.0d0
        hxyu(irg,itg,ipg) = gmxyu
        hxzu(irg,itg,ipg) = gmxzu
        hyyu(irg,itg,ipg) = gmyyu - 1.0d0
        hyzu(irg,itg,ipg) = gmyzu
        hzzu(irg,itg,ipg) = gmzzu - 1.0d0
!
      end do
    end do
  end do
!
end subroutine invhij_WL_export
