subroutine index_vec_down2up_export(hxxu,hxyu,hxzu,hyyu,hyzu,hzzu,vecxu,vecyu,veczu,vecxd,vecyd,veczd)
  use grid_parameter, only : nrg, ntg, npg
  implicit none
  real(8), pointer :: vecxu(:,:,:), vecyu(:,:,:), veczu(:,:,:), &
  &                   vecxd(:,:,:), vecyd(:,:,:), veczd(:,:,:), &
  &                   hxxu(:,:,:),  hxyu(:,:,:),  hxzu(:,:,:),  &
  &                   hyyu(:,:,:),  hyzu(:,:,:),  hzzu(:,:,:)
  real(8) :: gmxxu, gmxyu, gmxzu, gmyxu, gmyyu, gmyzu, &
  &          gmzxu, gmzyu, gmzzu
  integer :: ipg, itg, irg
!
!
  do ipg = 0, npg
    do itg = 0, ntg
      do irg = 0, nrg
        gmxxu = 1.0d0 + hxxu(irg,itg,ipg)
        gmxyu =         hxyu(irg,itg,ipg)
        gmxzu =         hxzu(irg,itg,ipg)
        gmyxu =         hxyu(irg,itg,ipg)
        gmyyu = 1.0d0 + hyyu(irg,itg,ipg)
        gmyzu =         hyzu(irg,itg,ipg)
        gmzxu =         hxzu(irg,itg,ipg)
        gmzyu =         hyzu(irg,itg,ipg)
        gmzzu = 1.0d0 + hzzu(irg,itg,ipg)
        vecxu(irg,itg,ipg) = gmxxu*vecxd(irg,itg,ipg) &
        &                  + gmxyu*vecyd(irg,itg,ipg) &
        &                  + gmxzu*veczd(irg,itg,ipg)
        vecyu(irg,itg,ipg) = gmyxu*vecxd(irg,itg,ipg) &
        &                  + gmyyu*vecyd(irg,itg,ipg) &
        &                  + gmyzu*veczd(irg,itg,ipg)
        veczu(irg,itg,ipg) = gmzxu*vecxd(irg,itg,ipg) &
        &                  + gmzyu*vecyd(irg,itg,ipg) &
        &                  + gmzzu*veczd(irg,itg,ipg)
      end do
    end do
  end do
!
end subroutine index_vec_down2up_export
