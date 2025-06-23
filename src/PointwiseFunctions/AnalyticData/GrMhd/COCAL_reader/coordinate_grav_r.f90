! r_coordinate for the field
!______________________________________________
module coordinate_grav_r
  use phys_constant, only : nnrg, long
  use grid_parameter, only : nrg, nrgin, rgin, rgmid, rgout, nrf, nrg_1, r_surf
  implicit none
  real(long)  :: rg(0:nnrg), rginv(0:nnrg) ! radial grid points.  inv means 1/r
  real(long)  :: hrg(nnrg), hrginv(nnrg)   ! mid points.
  real(long)  :: drg(nnrg), drginv(nnrg)   ! intervals.
contains
subroutine grid_r
! Grid points of r-coordinate
! --- Set up of meshes  ---  ( r(1) = dr ( not the center ) )
!
  implicit none
  real(long)  ::  drdr, drdrinv
  real(long) :: ratrr  ! ratio between subsequent step outside rgmid (or rather nrgin). \delta_{j+1} = k \delta_{j}
  real(long) :: rvdom, ratrrb, alge, dalge, error, rdet, rdetf
  real(long) :: rfdom, reso_min, drmin, drmininv, ratrf, ratrfb
  integer    :: nrout, ir, itry
!
  drdr =  1.0d0/dble(nrf)
  drdrinv = 1.0d0/drdr
  rvdom = rgout - rgmid
  nrout = nrg - nrgin
!
  drg(1:nrgin+1)    = drdr
  drginv(1:nrgin+1) = drdrinv
! 
  if (rgin.ge.1.0d-13.and.rgin.lt.1.0d0) then 
    rfdom = 1.0d0 - rgin
    ratrf = 0.1d0
    do 
      ratrfb = ratrf
      alge = -(ratrf**nrf - 1.0d0 - rfdom*drdrinv*(ratrf-1.0e0))
      dalge = dble(nrf)*ratrf**(nrf-1) - rfdom*drdrinv
      ratrf = ratrf + alge/dalge
      error = 2.d0*(ratrf - ratrfb)/(ratrf + ratrfb)
      if (abs(error) <= 1.0d-14) exit
    end do
    if (ratrf.gt.1.0d0) stop 'ratrf not good'
    drg(nrf) = drdr
    drginv(nrf) = 1.0d0/drg(nrf)
    do ir = nrf-1, 1, -1
      drg(ir) = drg(ir+1)*ratrf
      drginv(ir) = 1.0d0/drg(ir)
    end do
    do ir = nrf+1, nrgin
      drg(ir) = drg(nrf)
      drginv(ir) = 1.0d0/drg(ir)
    end do
  end if
!
do itry = 0, 3 
  if (itry.eq.0) ratrr = 2.0e-01   ! good for finer grid
  if (itry.eq.1) ratrr = 2.0e0   ! good for finer grid
  if (itry.eq.2) ratrr = 10.0e0  ! good for coarser grid
  if (itry.eq.3) ratrr = 50.0e0  ! good for coarsest grid
! 
  DO
    ratrrb = ratrr
    alge = -(ratrr**(nrout+1)-ratrr-rvdom*drdrinv*(ratrr-1.0e0))
    dalge = dble(nrout+1)*ratrr**nrout - 1.0e0 - rvdom*drdrinv
    ratrr = ratrr + alge/dalge
    error = 2.d0*(ratrr - ratrrb)/(ratrr + ratrrb)
! 
    IF (abs(error)<=1.d-14) EXIT 
  END DO
! r-coordinate. interval dr
  DO ir = nrgin+1, nrg
    drg(ir) = ratrr*drg(ir-1)
    drginv(ir) = 1.0e0/drg(ir)
  END DO
! r-coordinate. grid ponit and mid point.
  rg(0) = rgin
  if (rg(0).le.1.0d-14) then 
    rginv(0) = 0.0e0
  else 
    rginv(0) = 1.0d0/rg(0)
  end if
  DO ir = 1, nrg
    rg(ir) = rg(ir-1) + drg(ir)
    hrg(ir) = (rg(ir) + rg(ir-1))*0.5e0
    rginv(ir) = 1.0e0/rg(ir)
    hrginv(ir) = 1.0e0/hrg(ir)
!      write(6,*)ir,rg(ir)
  END DO
!      WRITE(6,'(1p,2e12.4)') error, ratrr
!      WRITE(6,'(1p,2e12.4)') rgout, rg(nrg)
  rdet  = (rg(nrg) - rgout)/rgout
  rdetf = 0.0d0
  if (rgin.lt.1.0d0) rdetf =  rg(nrf) - 1.0d0
  IF(dabs(rdet)< 1.e-10) exit
  IF(dabs(rdet)>=1.e-10)then
    WRITE(6,*) ' bad coordinate itry =', itry
    WRITE(6,*) ' bad coordinate GR ', rdet, ratrr
    WRITE(6,*) ' bad coordinate FLUID ', rdetf, ratrf
    if (itry.eq.3) STOP
  END IF
end do
end subroutine grid_r
!
subroutine grid_r_bhex(char_bh)
! Grid points of r-coordinate
! --- Set up of meshes  ---  ( r(1) = dr ( not the center ) )
!
  use def_bh_parameter, only : mass_pBH
  implicit none
  real(long)  ::  drdr, drdrinv
  real(long) :: ratrr  ! ratio between subsequent step outside rgmid (or rather nrgin). \delta_{j+1} = k \delta_{j}
  real(long) :: rvdom, ratrrb, alge, dalge, error, rdet, rdetf
  real(long) :: rfdom, reso_min, drmin, drmininv, ratrf, ratrfb
  integer    :: nrout, ir, itry, nrf_bh
  real(long) :: dr1, dr1inv, tolerant_val = 5.0d-14
  integer    :: nr_count, count
  character(len=3) :: char_bh
!
  drdr =  1.0d0/dble(nrf)
  drdrinv = 1.0d0/drdr
!
  drg(1:nrgin+1)    = drdr
  drginv(1:nrgin+1) = drdrinv
! 
!  if (rgin.ge.1.0d-13.and.rgin.lt.1.0d0) then 
  if (rgin.lt.1.0d0) then 
!
do itry = 0, 3 
  if (itry.eq.0) ratrr = 1.01d0 ! good for finer grid
  if (itry.eq.1) ratrr = 1.1d0   ! good for finer grid
  if (itry.eq.2) ratrr = 2.0d0  ! good for coarser grid
  if (itry.eq.3) ratrr = 10.0d0  ! good for coarsest grid
! 
!    dr1 = rgin/(0.5d0*dble(nrf))
    dr1 = rgin/(0.75d0*dble(nrf))
    if (char_bh.eq.'pBH') dr1 = 0.5d0*mass_pBH/(0.75d0*dble(nrf))
    dr1inv = 1.0d0/dr1
    rvdom = 1.0d0 - rgin
    nrout = nrf
!
  count = 0
  DO
    count = count + 1
    ratrrb = ratrr
    alge = -(ratrr**(nrout)-1.0d0-rvdom*dr1inv*(ratrr-1.0d0))
    dalge = dble(nrout)*ratrr**(nrout-1) - rvdom*dr1inv
!test    alge = -((ratrr**nrout-1.0d0)/(ratrr-1.0d0)-rvdom*dr1inv)
!test    dalge = (dble(nrout-1)*ratrr**nrout-dble(nrout)*ratrr**(nrout-1)+1.0d0) &
!test    &     / (ratrr-1.0d0)**2
    ratrr = ratrr + alge/dalge
    error = 2.d0*(ratrr - ratrrb)/(ratrr + ratrrb)
! 
    IF (abs(error)<=1.d-15) EXIT 
    IF (count.ge.1000) exit
  END DO
!test  DO
!test    count = count + 1
!test    ratrrb = ratrr
!test    alge = -(dlog(ratrr**nrout-1.0d0)-dlog(ratrr-1.0d0)-dlog(rvdom*dr1inv))
!test    dalge = dble(nrout)*ratrr**(nrout-1)/(ratrr**nrout-1.0d0) &
!test    &     - 1.0d0/(ratrr-1.0d0)
!test    ratrr = ratrr + alge/dalge
!test    error = 2.d0*(ratrr - ratrrb)/(ratrr + ratrrb)
!test! 
!test    IF (abs(error)<=1.d-15) EXIT 
!test    IF (count.ge.1000) exit
!test  END DO
!
  IF (count.ge.1000) cycle
! r-coordinate. interval dr
    drg(1) = dr1
    drginv(1) = 1.0d0/dr1
    do ir = 2, nrf
      drg(ir) = ratrr*drg(ir-1)
      drginv(ir) = 1.0d0/drg(ir)
    end do
!
    rg(0) = rgin
    DO ir = 1, nrf
      rg(ir) = rg(ir-1) + drg(ir)
    end do
!
  rdet  = (rg(nrf) - 1.0d0)/1.0d0
  rdetf = 0.0d0
  if (rgin.lt.1.0d0) rdetf =  rg(nrf) - 1.0d0
  IF(dabs(rdet)< 2.e-14) then
    WRITE(6,*) 'good coordinate itry =', itry
    WRITE(6,*) 'good coordinate GR ', rdet, ratrr
    WRITE(6,*) 'good coordinate FLUID ', rdetf, ratrf
    exit
  end if
  IF(dabs(rdet)>=2.e-14)then
    WRITE(6,*) ' bad coordinate itry =', itry
    WRITE(6,*) ' bad coordinate GR ', rdet, ratrr
    WRITE(6,*) ' bad coordinate FLUID ', rdetf, ratrf
    if (itry.eq.3) STOP
  END IF
!
end do
!
! region III
    drdr =  drg(nrgin)
    drdrinv = 1.0d0/drdr
    rvdom = 2.0d0*rgmid
    nrout = nrf
    ratrr = 2.0d0
!
    DO
      ratrrb = ratrr
      alge = -(ratrr**(nrout+1)-ratrr-rvdom*drdrinv*(ratrr-1.0d0))
      dalge = dble(nrout+1)*ratrr**nrout - 1.0d0 - rvdom*drdrinv
      ratrr = ratrr + alge/dalge
      error = 2.d0*(ratrr - ratrrb)/(ratrr + ratrrb)
! 
      IF (abs(error)<=1.d-15) EXIT 
    END DO
! r-coordinate. interval dr
    DO ir = nrgin+1, nrgin+nrf
      drg(ir) = ratrr*drg(ir-1)
      drginv(ir) = 1.0d0/drg(ir)
    END DO
  end if
!
do itry = 0, 3 
  if (itry.eq.0) ratrr = 1.05d-00 ! good for finer grid
  if (itry.eq.1) ratrr = 2.0d0   ! good for finer grid
  if (itry.eq.2) ratrr = 10.0d0  ! good for coarser grid
  if (itry.eq.3) ratrr = 50.0d0  ! good for coarsest grid
! 
  drdr =  1.0d0/dble(nrf)
  drdrinv = 1.0d0/drdr
  rvdom = rgout - rgmid
  nrout = nrg - nrgin
!  if (rgin.ge.1.0d-13.and.rgin.lt.1.0d0) then 
  if (rgin.lt.1.0d0) then 
    drdr =  drg(nrgin+nrf)
    drdrinv = 1.0d0/drdr
    rvdom = rgout - 3.0d0*rgmid
    nrout = nrg - (nrgin + nrf)
  end if
!
  DO
    ratrrb = ratrr
!test    alge = -(ratrr**(nrout+1)-ratrr-rvdom*drdrinv*(ratrr-1.0d0))
!test    dalge = dble(nrout+1)*ratrr**nrout - 1.0d0 - rvdom*drdrinv
    alge = -(dble(nrout+1)*log(ratrr)-log(ratrr+rvdom*drdrinv*(ratrr-1.0d0)))
    dalge = dble(nrout+1)/ratrr - (1.0d0+rvdom*drdrinv)/ &
    &                             (ratrr+rvdom*drdrinv*(ratrr-1.0d0))
    ratrr = ratrr + alge/dalge
    error = 2.d0*(ratrr - ratrrb)/(ratrr + ratrrb)
! 
    IF (abs(error)<=1.d-16) EXIT 
  END DO
!
! r-coordinate. interval dr
  nr_count = nrgin+1
!  if (rgin.ge.1.0d-13.and.rgin.lt.1.0d0) nr_count = nrgin+nrf+1
  if (rgin.lt.1.0d0) nr_count = nrgin+nrf+1
  DO ir = nr_count, nrg
    drg(ir) = ratrr*drg(ir-1)
    drginv(ir) = 1.0d0/drg(ir)
  END DO
! r-coordinate. grid ponit and mid point.
  rg(0) = rgin
  if (rg(0).le.1.0d-14) then 
    rginv(0) = 0.0d0
  else 
    rginv(0) = 1.0d0/rg(0)
  end if
  DO ir = 1, nrg
    rg(ir) = rg(ir-1) + drg(ir)
    hrg(ir) = (rg(ir) + rg(ir-1))*0.5d0
    rginv(ir) = 1.0d0/rg(ir)
    hrginv(ir) = 1.0d0/hrg(ir)
!      write(6,*)ir,rg(ir)
  END DO
!      WRITE(6,'(1p,2e12.4)') error, ratrr
!      WRITE(6,'(1p,2e12.4)') rgout, rg(nrg)
  rdet  = (rg(nrg) - rgout)/rgout
  rdetf = 0.0d0
  if (rgin.lt.1.0d0) rdetf =  rg(nrf) - 1.0d0
  IF(dabs(rdet)< tolerant_val) then
    WRITE(6,*) 'good coordinate itry =', itry
    WRITE(6,*) 'good coordinate GR ', rdet, ratrr
    WRITE(6,*) 'good coordinate FLUID ', rdetf, ratrf
    exit
  end if
  IF(dabs(rdet)>=tolerant_val)then
    WRITE(6,*) ' bad coordinate itry =', itry
    WRITE(6,*) ' bad coordinate GR ', rdet, ratrr
    WRITE(6,*) ' bad coordinate FLUID ', rdetf, ratrf
    if (itry.eq.3) STOP
  END IF
end do
! testtest
!do ir = 0, nrg
!write(6,*) rg(ir)
!end do
!stop
!
end subroutine grid_r_bhex
!
!
subroutine grid_r_bns

  implicit none
  real(long)  ::  drdr, drdrinv
  real(long) :: ratrr  ! ratio between subsequent step outside rgmid (or rather nrgin). \delta_{j+1} = k \delta_{j}
  real(long) :: rvdom, ratrrb, alge, dalge, error, rdet, rdetf, rdet1
  real(long) :: rfdom, reso_min, drmin, drmininv, ratrf, ratrfb
  integer    :: nrout, ir, itry, nrf_bh
  real(long) :: dr1, dr1inv, tolerant_val = 5.0d-14
  real(long) :: dr2, dr2inv
  integer    :: nr_count, count, nr3
  character(len=3) :: char_bh

!
! Region I r<=rgmid  for mpt3.
  drdr =  1.0d0/dble(nrf)
  drdrinv = 1.0d0/drdr
!
  drg(1:nrgin+1)    = drdr
  drginv(1:nrgin+1) = drdrinv
!
! the following if block only for mpt1,2:   Sets up the grid for r<3rgmid
! for mpt3 we have only the above lines i.e.  r<rgmid
!
  if (rgin==0.0d0) then
    write(6,*) "REGION S, I, II, III:"
!
    drdr = r_surf/dble(nrf)
    drdrinv = 1.0d0/drdr

!   Region S
    drg(1:nrf)    = drdr
    drginv(1:nrf) = drdrinv

!   Region II
    dr2 =  1.0e0/dble(nrg_1)            ! nrg_1 plays the role of nrf in BBH
    drg(nrg_1+1:nrgin)    = dr2
    drginv(nrg_1+1:nrgin) = 1.0e0/dr2

    if(r_surf<1.0d0) then
      if(nrf==nrg_1) then
        write(6,*) "nrf==nrg_1 but r_surf<1.0...exiting"
        stop
      end if
      do itry = 0, 1
        if (itry.eq.0) ratrr = 2.0e-01 ! good for finer grid
        if (itry.eq.1) ratrr = 2.0e0   ! good for finer grid
!
        dr1 = drg(nrf)                 ! drg(nrf+1)= k drg(nrf)
        dr1inv = 1.0d0/dr1
        rvdom = 1.0d0 - r_surf
        nrout = nrg_1 - nrf

        DO
          ratrrb = ratrr
          alge = -(ratrr**(nrout+1)-ratrr-rvdom*dr1inv*(ratrr-1.0e0))
          dalge = dble(nrout+1)*ratrr**nrout - 1.0e0 - rvdom*dr1inv
          ratrr = ratrr + alge/dalge
          error = 2.d0*(ratrr - ratrrb)/(ratrr + ratrrb)
!
          IF (abs(error)<=1.d-14) EXIT
        END DO
        write(6,*) 'region I: ratrr=', ratrr
!
!       r-coordinate. interval dr
        do ir = nrf+1, nrg_1
          drg(ir) = ratrr*drg(ir-1)
          drginv(ir) = 1.0e0/drg(ir)
        end do

        rg(0) = rgin
        DO ir = 1, nrgin
          rg(ir) = rg(ir-1) + drg(ir)
!          write(6,*) ir, rg(ir), drg(ir)
        end do
!
        rdet  = (rg(nrgin) - rgmid)/rgmid
        rdetf = 0.0d0
        if (rgin.lt.1.0d0) rdet1 =  rg(nrg_1) - 1.0d0
        if (rgin.lt.1.0d0) rdetf =  rg(nrf) - r_surf
        IF(dabs(rdet)< 1.e-10) then
          WRITE(6,*) 'good coordinate itry =', itry
          WRITE(6,*) 'good coordinate GR : rdet1, rdet, ratrr ', rdet1, rdet, ratrr
          WRITE(6,*) 'good coordinate FLUID : rdetf', rdetf
          exit
        end if
        IF(dabs(rdet)>=1.e-10)then
          WRITE(6,*) ' bad coordinate itry =', itry
          WRITE(6,*) ' bad coordinate GR : rdet1, rdet, ratrr ', rdet1, rdet, ratrr
          WRITE(6,*) ' bad coordinate FLUID : rdetf ', rdetf
          if (itry.eq.1) STOP
        END IF
      end do
    else
      write(6,*) "Region S = Region I"
      if(nrf.ne.nrg_1)  then
        write(6,*) "nrf is not equal to nrg_1, but r_surf=1...exiting"
        stop
      end if
    end if                   !  r_surf < 1.0
!
!   region III.
    drdr =  drg(nrgin)       !  In bhex first interval in region III: drg(nrgin+1)= k drg(nrgin)
                             !  If we want to change drdr we must also modify
                             !  LINE (*)
    drdrinv = 1.0d0/drdr
    rvdom = 2.0d0*rgmid
    nr3 = nrg_1               !  plays the role of nrf in BBH
    ratrr = 2.0e0
!
    DO
      ratrrb = ratrr
      alge = -(ratrr**(nr3+1)-ratrr-rvdom*drdrinv*(ratrr-1.0e0))
      dalge = dble(nr3+1)*ratrr**nr3 - 1.0e0 - rvdom*drdrinv
      ratrr = ratrr + alge/dalge
      error = 2.d0*(ratrr - ratrrb)/(ratrr + ratrrb)
!
      IF (abs(error)<=1.d-14) EXIT
    END DO

    write(6,*) 'region III: ratrr=', ratrr, "    dr1=", drdr, "    dr(nrf)=", drg(nrf)

!   r-coordinate. interval dr:   drg(nrgin+1)= k drdr
    drg(nrgin+1)    = ratrr*drdr
    drginv(nrgin+1) = 1.0e0/drg(nrgin+1)

    DO ir = nrgin+2, nrgin+nr3
      drg(ir) = ratrr*drg(ir-1)               !   LINE (*)
      drginv(ir) = 1.0e0/drg(ir)
    END DO
!
    rg(0) = rgin
    DO ir = 1, nrgin+nr3
      rg(ir) = rg(ir-1) + drg(ir)
    end do
    write(6,*)  "rg(nrgin+nr3)=", rg(nrgin+nr3), "      4rgmid=", 4.0d0*rgmid
  end if                         !  rgin==0
!
!  Now for r > 3rgmid if mpt1,2
!  or      r > rgmid  if mpt3
!
do itry = 0, 3
  if (itry.eq.0) ratrr = 2.0e-01 ! good for finer grid
  if (itry.eq.1) ratrr = 2.0e0   ! good for finer grid
  if (itry.eq.2) ratrr = 10.0e0  ! good for coarser grid
  if (itry.eq.3) ratrr = 50.0e0  ! good for coarsest grid
!
  drdr =  1.0d0/dble(nrf)
  drdrinv = 1.0d0/drdr
  rvdom = rgout - rgmid
  nrout = nrg - nrgin
  if (rgin==0.0d0) then
    drdr =  drg(nrgin+nr3)
    drdrinv = 1.0d0/drdr
    rvdom = rgout - 3.0d0*rgmid
    nrout = nrg - (nrgin + nr3)
  end if
!
  DO
    ratrrb = ratrr
    alge = -(ratrr**(nrout+1)-ratrr-rvdom*drdrinv*(ratrr-1.0e0))
    dalge = dble(nrout+1)*ratrr**nrout - 1.0e0 - rvdom*drdrinv
    ratrr = ratrr + alge/dalge
    error = 2.d0*(ratrr - ratrrb)/(ratrr + ratrrb)
!
    IF (abs(error)<=1.d-14) EXIT
  END DO

  write(6,*) 'region IV: ratrr=', ratrr

! r-coordinate. interval dr
  nr_count = nrgin+1
  if (rgin==0.0d0) nr_count = nrgin+nr3+1
  DO ir = nr_count, nrg
    drg(ir) = ratrr*drg(ir-1)
    drginv(ir) = 1.0e0/drg(ir)
  END DO
! r-coordinate. grid ponit and mid point.
  rg(0) = rgin
  if (rg(0).le.1.0d-14) then
    rginv(0) = 0.0e0
  else
    rginv(0) = 1.0d0/rg(0)
  end if
  DO ir = 1, nrg
    rg(ir) = rg(ir-1) + drg(ir)
    hrg(ir) = (rg(ir) + rg(ir-1))*0.5e0
    rginv(ir) = 1.0e0/rg(ir)
    hrginv(ir) = 1.0e0/hrg(ir)
!      write(6,*)ir,rg(ir)
  END DO
!      WRITE(6,'(1p,2e12.4)') error, ratrr
!      WRITE(6,'(1p,2e12.4)') rgout, rg(nrg)
  rdet  = (rg(nrg) - rgout)/rgout
  rdetf = 0.0d0
  if (rgin.lt.1.0d0) rdet1 =  rg(nrg_1) - 1.0d0
  if (rgin.lt.1.0d0) rdetf =  rg(nrf) - r_surf
  IF(dabs(rdet)< 1.e-10) then
    WRITE(6,*) 'good coordinate itry =', itry
    WRITE(6,*) 'good coordinate GR : rdet1, rdet, ratrr ', rdet1, rdet, ratrr
    WRITE(6,*) 'good coordinate FLUID : rdetf ', rdetf
    exit
  end if
  IF(dabs(rdet)>=1.e-10)then
    WRITE(6,*) ' bad coordinate itry =', itry
    WRITE(6,*) ' bad coordinate GR : rdet1, rdet, ratrr', rdet1, rdet, ratrr
    WRITE(6,*) ' bad coordinate FLUID : rdetf ', rdetf
    if (itry.eq.3) STOP
  END IF
end do
!
end subroutine grid_r_bns
!
!
subroutine grid_r_bns_const

  implicit none
  real(long)  ::  drdr, drdrinv
  real(long) :: ratrr  ! ratio between subsequent step outside rgmid (or rather nrgin). \delta_{j+1} = k \delta_{j}
  real(long) :: rvdom, ratrrb, alge, dalge, error, rdet, rdetf, rdet1
  real(long) :: rfdom, reso_min, drmin, drmininv, ratrf, ratrfb
  integer    :: nrout, ir, itry, nrconst
  real(long) :: dr1, dr1inv, tolerant_val = 5.0d-14
  real(long) :: dr2, dr2inv, rgconst
  integer    :: nr_count, count, nr3
  character(len=3) :: char_bh

!
  if (r_surf .ne. 1.0) then
    write(6,*) "r_surf is not equal to 1.0....resetting r_surf=1"
    r_surf = 1.0
  end if
!
! Region I r<=rgmid  for mpt3.
  drdr =  1.0d0/dble(nrf)
  drdrinv = 1.0d0/drdr
!
  drg(1:nrgin+1)    = drdr
  drginv(1:nrgin+1) = drdrinv
!
! the following if block only for mpt1,2: 
! for mpt3 we have only the above lines i.e.  r<rgmid
!
  if (rgin==0.0d0) then
!   ********* Calc rgmid based on nrgin and nrf *******************
    rgmid = nrgin*drdr
!   
!    rgconst = 2*rgmid + 2      ! Corresponds to rgmid = 3.0
    rgconst = 2*3.0 + 2      ! Corresponds to rgmid = 3.0
    rvdom = rgout - rgconst
!    nrconst = 2*nrgin + 2*nrf
    nrconst = 256            ! Corresponds to nrf=32  nrgin=96   2*nrf+2*nrgin
    nrout = nrg - nrconst
!
    if (rgmid>3.0)  then
      write(6,*) "rgmid must be less than 3.0....exiting"
      stop
    end if

!   r-coordinate. interval dr
    DO ir = 1, nrconst
      drg(ir) = drdr
      drginv(ir) = 1.0e0/drg(ir)
    END DO
!
    rg(0) = rgin
    DO ir = 1, nrconst
      rg(ir) = rg(ir-1) + drg(ir)
!      write(6,*) ir, rg(ir), drg(ir)
    end do
  end if
!
!  Now for r > 3rgmid if mpt1,2
!  or      r > rgmid  if mpt3
!
do itry = 0, 3
  if (itry.eq.0) ratrr = 2.0e-01 ! good for finer grid
  if (itry.eq.1) ratrr = 2.0e0   ! good for finer grid
  if (itry.eq.2) ratrr = 10.0e0  ! good for coarser grid
  if (itry.eq.3) ratrr = 50.0e0  ! good for coarsest grid
!
  drdr =  1.0d0/dble(nrf)
  drdrinv = 1.0d0/drdr
  rvdom = rgout - rgmid
  nrout = nrg - nrgin
  if (rgin==0.0d0) then
    drdr = drg(nrconst)
    drdrinv = 1.0d0/drdr
    rvdom = rgout - rgconst
    nrout = nrg - nrconst
  end if
!
  DO
    ratrrb = ratrr
    alge = -(ratrr**(nrout+1)-ratrr-rvdom*drdrinv*(ratrr-1.0e0))
    dalge = dble(nrout+1)*ratrr**nrout - 1.0e0 - rvdom*drdrinv
    ratrr = ratrr + alge/dalge
    error = 2.d0*(ratrr - ratrrb)/(ratrr + ratrrb)
!
    IF (abs(error)<=1.d-14) EXIT
  END DO

  write(6,*) 'region II: ratrr=', ratrr

! r-coordinate. interval dr
  nr_count = nrgin+1
  if (rgin==0.0d0) nr_count = nrconst+1
  DO ir = nr_count, nrg
    drg(ir) = ratrr*drg(ir-1)
    drginv(ir) = 1.0e0/drg(ir)
  END DO
! r-coordinate. grid ponit and mid point.
  rg(0) = rgin
  if (rg(0).le.1.0d-14) then
    rginv(0) = 0.0e0
  else
    rginv(0) = 1.0d0/rg(0)
  end if
  DO ir = 1, nrg
    rg(ir) = rg(ir-1) + drg(ir)
    hrg(ir) = (rg(ir) + rg(ir-1))*0.5e0
    rginv(ir) = 1.0e0/rg(ir)
    hrginv(ir) = 1.0e0/hrg(ir)
!    write(6,*)  ir,rg(ir), drg(ir)
  END DO
!
!
  rdet  = dabs(rg(nrg) - rgout)/rgout
  rdetf = 0.0d0
  if (rgin.lt.1.0d0) rdetf =  rg(nrf) - 1.0d0
  IF(rdet< 1.e-10) then
    WRITE(6,*) 'good coordinate itry =', itry
    WRITE(6,*) 'good coordinate GR : rdet, ratrr ', rdet, ratrr
    WRITE(6,*) 'good coordinate FLUID : rdetf, ratrf', rdetf, ratrf
    if (rg(0)==0.0) then
      write(6,*) "Number of constant intervals, dr, N*dr :", nrconst, drdr, nrconst*drdr
      write(6,*) "Number of non-constant intervals, rgout:", nrg - nrconst, rgout
    else
      write(6,*) "Number of constant intervals, dr, N*dr :", nrgin, drdr, nrgin*drdr 
      write(6,*) "Number of non-constant intervals, rgout:", nrg - nrgin, rgout
    end if
    exit
  end if
  IF(rdet>=1.e-10)then
    WRITE(6,*) ' bad coordinate itry =', itry
    WRITE(6,*) ' bad coordinate GR : rdet, ratrr', rdet, ratrr
    WRITE(6,*) ' bad coordinate FLUID : rdetf, ratrf', rdetf, ratrf
    if (itry.eq.3) STOP
  END IF
end do
!
end subroutine grid_r_bns_const
!
!
subroutine grid_r_bqs

  implicit none
  real(long)  ::  drdr, drdrinv
  real(long) :: ratrr  ! ratio between subsequent step outside rgmid (or rather nrgin). \delta_{j+1} = k \delta_{j}
  real(long) :: rvdom, ratrrb, alge, dalge, error, rdet, rdetf, rdet1
  real(long) :: rfdom, reso_min, drmin, drmininv, ratrf, ratrfb
  real(long) :: rr1, rr2, rr3, dr3
  integer    :: nrout, ir, itry, nrconst
  real(long) :: dr1, dr1inv, tolerant_val = 5.0d-14
  real(long) :: dr2, dr2inv, rgconst
  integer    :: nr_count, count, nr3, nr2, nr1
  character(len=3) :: char_bh

!
  if (r_surf .ne. 1.0) then
    write(6,*) "r_surf is not equal to 1.0  ", r_surf, "....resetting r_surf=1"
    r_surf = 1.0
  end if
!
  if (rgin>0.0) then
    ! Region I r<=rgmid  for mpt3.
    drdr =  1.0d0/dble(nrf)
    drdrinv = 1.0d0/drdr
!
    drg(1:nrgin+1)    = drdr
    drginv(1:nrgin+1) = drdrinv
!
    do itry = 0, 3
      if (itry.eq.0) ratrr = 2.0e-01 ! good for finer grid
      if (itry.eq.1) ratrr = 2.0e0   ! good for finer grid
      if (itry.eq.2) ratrr = 10.0e0  ! good for coarser grid
      if (itry.eq.3) ratrr = 50.0e0  ! good for coarsest grid
    !
      drdr =  1.0d0/dble(nrf)
      drdrinv = 1.0d0/drdr
      rvdom = rgout - rgmid
      nrout = nrg - nrgin
    
      DO
        ratrrb = ratrr
        alge = -(ratrr**(nrout+1)-ratrr-rvdom*drdrinv*(ratrr-1.0e0))
        dalge = dble(nrout+1)*ratrr**nrout - 1.0e0 - rvdom*drdrinv
        ratrr = ratrr + alge/dalge
        error = 2.d0*(ratrr - ratrrb)/(ratrr + ratrrb)
    !
        IF (abs(error)<=1.d-14) EXIT
      END DO
    
      write(6,*) 'region II: ratrr=', ratrr
    
    ! r-coordinate. interval dr
      nr_count = nrgin+1
      DO ir = nr_count, nrg
        drg(ir) = ratrr*drg(ir-1)
        drginv(ir) = 1.0e0/drg(ir)
      END DO
    ! r-coordinate. grid ponit and mid point.
      rg(0) = rgin
      if (rg(0).le.1.0d-14) then
        rginv(0) = 0.0e0
      else
        rginv(0) = 1.0d0/rg(0)
      end if
      DO ir = 1, nrg
        rg(ir) = rg(ir-1) + drg(ir)
        hrg(ir) = (rg(ir) + rg(ir-1))*0.5e0
        rginv(ir) = 1.0e0/rg(ir)
        hrginv(ir) = 1.0e0/hrg(ir)
    !    write(6,*)  ir,rg(ir), drg(ir)
      END DO
    !
    !
      rdet  = dabs(rg(nrg) - rgout)/rgout
      rdetf = 0.0d0
      IF(rdet< 1.e-10) then
        WRITE(6,*) 'good coordinate itry =', itry
        WRITE(6,*) 'good coordinate GR : rdet, ratrr ', rdet, ratrr
        WRITE(6,*) 'good coordinate FLUID : rdetf, ratrf', rdetf, ratrf
        exit
      end if
      IF(rdet>=1.e-10)then
        WRITE(6,*) ' bad coordinate itry =', itry
        WRITE(6,*) ' bad coordinate GR : rdet, ratrr', rdet, ratrr
        WRITE(6,*) ' bad coordinate FLUID : rdetf, ratrf', rdetf, ratrf
        if (itry.eq.3) STOP
      END IF
    end do
! 
  else
    drdr =  1.0d0/dble(nrf)
    drdrinv = 1.0d0/drdr
!
    nr1 = nrf + int((1.25d0-1.0d0)/drdr)    ! Up until r=1.25  dr=1/nrf

    drg(1:nr1)    = drdr
    drginv(1:nr1) = drdrinv
!
    dr2 = 2.0d0*drdr     !    Choose dr2 from r=1.25 up to r=5.  2*drdr is ok for nrf=100
    !dr2 = 2.56d0*drdr     !    Choose dr2 from r=1.25 up to r=5   2.56*drdr is ok for nrf=128
    nr2 = nr1 + int((5.0d0-1.25d0)/dr2)
!    if (nr2 > 330)  then
!      write(6,*)  "nr2 = ", nr2, "   must be less than 330. Exiting..."
!      stop
!    end if

!   r-coordinate. interval dr
    DO ir = nr1+1, nr2
      drg(ir) = dr2
      drginv(ir) = 1.0e0/drg(ir)
    END DO
!
    rg(0) = rgin
    DO ir = 1, nr2
      rg(ir) = rg(ir-1) + drg(ir)
!      write(6,*) "AAAAA", ir, rg(ir), drg(ir)
    end do

!   Find rgmid
    DO ir = 1, nr2
      if ((rg(ir)-1.0d-10)<=rgmid .and. rgmid<(rg(ir+1)-1.0d-10)) then
        if (dabs(rg(ir)-rgmid)<=1.0d-14) then
          nrgin = ir
          exit
        else
          drg(ir+1) = rgmid - rg(ir)
          drg(ir+2) = rg(ir+1) - rgmid
          nrgin = ir + 1 
          drg(ir+3:nr2) = dr2
          exit
        end if
      end if
    end do
!
    rg(0) = rgin
    DO ir = 1, nr2
      rg(ir) = rg(ir-1) + drg(ir)
!      write(6,*) "BBBBB", ir, rg(ir), drg(ir)
    end do
    write(6,*) "@@@@@ nrgin = ", nrgin, "   and rg(nrgin) = ", rg(nrgin)
!
    do itry = 0, 3
      if (itry.eq.0) ratrr = 2.0e-01 ! good for finer grid
      if (itry.eq.1) ratrr = 2.0e0   ! good for finer grid
      if (itry.eq.2) ratrr = 10.0e0  ! good for coarser grid
      if (itry.eq.3) ratrr = 50.0e0  ! good for coarsest grid
    !
      drdr = drg(nr2)
      drdrinv = 1.0d0/dr2
      rvdom = rgout - rg(nr2)   
      nrout = nrg - nr2
    !
      DO
        ratrrb = ratrr
        alge = -(ratrr**(nrout+1)-ratrr-rvdom*drdrinv*(ratrr-1.0e0))
        dalge = dble(nrout+1)*ratrr**nrout - 1.0e0 - rvdom*drdrinv
        ratrr = ratrr + alge/dalge
        error = 2.d0*(ratrr - ratrrb)/(ratrr + ratrrb)
    !
        IF (abs(error)<=1.d-14) EXIT
      END DO
    
      write(6,*) 'region II: ratrr=', ratrr
    
    ! r-coordinate. interval dr
      nr_count = nr2+1
      DO ir = nr_count, nrg
        drg(ir) = ratrr*drg(ir-1)
        drginv(ir) = 1.0e0/drg(ir)
      END DO
    ! r-coordinate. grid ponit and mid point.
      rg(0) = rgin
      if (rg(0).le.1.0d-14) then
        rginv(0) = 0.0e0
      else
        rginv(0) = 1.0d0/rg(0)
      end if
      DO ir = 1, nrg
        rg(ir) = rg(ir-1) + drg(ir)
        hrg(ir) = (rg(ir) + rg(ir-1))*0.5e0
        rginv(ir) = 1.0e0/rg(ir)
        hrginv(ir) = 1.0e0/hrg(ir)
!        write(6,*)  ir,rg(ir), drg(ir)
      END DO
    !
    !
      rdet  = dabs(rg(nrg) - rgout)/rgout
      rdetf = 0.0d0
      if (rgin.lt.1.0d0) rdetf =  rg(nrf) - 1.0d0
      IF(rdet< 1.e-10) then
        WRITE(6,*) 'good coordinate itry =', itry
        WRITE(6,*) 'good coordinate GR : rdet, ratrr ', rdet, ratrr
        WRITE(6,*) 'good coordinate FLUID : rdetf, ratrf', rdetf, ratrf
        exit
      end if
      IF(rdet>=1.e-10)then
        WRITE(6,*) ' bad coordinate itry =', itry
        WRITE(6,*) ' bad coordinate GR : rdet, ratrr', rdet, ratrr
        WRITE(6,*) ' bad coordinate FLUID : rdetf, ratrf', rdetf, ratrf
        if (itry.eq.3) STOP
      END IF
    end do
  end if
!
end subroutine grid_r_bqs
!

!
subroutine grid_r_bht(char_bh)
  use def_bh_parameter, only : mass_pBH
  use def_bh_parameter, only : mass_bh, spin_bh 
  implicit none
  real(long)  ::  drdr, drdrinv
  real(long) :: ratrr  ! ratio between subsequent step outside rgmid (or rather nrgin). \delta_{j+1} = k \delta_{j}
  real(long) :: rvdom, ratrrb, alge, dalge, error, rdet, rdetf
  real(long) :: rfdom, reso_min, drmin, drmininv, ratrf, ratrfb
  integer    :: nrout, ir, itry, nrf_bh
  real(long) :: dr_inh, rh, dr1, dr1inv, tolerant_val = 5.0d-14
  integer    :: nr_count, count, nr_inh
  character(len=3) :: char_bh
!
!***********************************************************************************
!*                                                                                 *
!*    rgin   : Black hole radius.                                                  *
!*    r_surf : Inner radius of torus.                                              *
!*    nrgin  : Number of intervals between rgin and r_surf.                        *
!*             The r-grid spacing there in non-equidistant                         *
!*    rgmid  : Does not correspond to anything particular, but is close to         *
!*             the outer boundary of torus, which varies with the solution.        *
!*             The r-grid is equidistant from r_surf to rgmid.                     *
!*    nrf    : Number of intervals from rgin to rgmid (nrf>nrgin).                 *
!*                                                                                 *
!***********************************************************************************
!
  if(rgmid<r_surf)   STOP "rgmid should be greater than r_surf."
  if(nrgin>nrf)      STOP "nrgin should be smaller than nrf."

  drdr = (rgmid - r_surf)/(dble(nrf) - dble(nrgin))
  drdrinv = 1.0d0/drdr
!  write(6,'(a10,1p,e23.15)') "drdr=", drdr
!
  drg(1:nrf+1)    = drdr
  drginv(1:nrf+1) = drdrinv
!
!-------------------------------------------------------
! Inside the horizon
  rh = mass_bh + dsqrt(mass_bh*mass_bh-spin_bh*spin_bh)
  nr_inh = 8

!  dr1 = rgin/dble(nrgin)
!    dr1 = mass_bh/dble(nrf)
  dr_inh = (rh - rgin)/nr_inh
  drg(1:nr_inh)    = dr_inh
  drginv(1:nr_inh) = 1.0d0/dr_inh
  rg(0) = rgin
  DO ir = 1, nr_inh
    rg(ir) = rg(ir-1) + drg(ir)
  end do

!-------------------------------------------------------
! Outside the horizon 
  dr1 = dr_inh
  do itry = 0, 3 
    if (itry.eq.0) ratrr = 1.01d0 ! good for finer grid
    if (itry.eq.1) ratrr = 1.1d0   ! good for finer grid
    if (itry.eq.2) ratrr = 2.0d0  ! good for coarser grid
    if (itry.eq.3) ratrr = 10.0d0  ! good for coarsest grid
! 
    if (char_bh.eq.'pBH') dr1 = 0.5d0*mass_pBH/(0.75d0*dble(nrf))
    dr1inv = 1.0d0/dr1
    rvdom = r_surf - rh
    nrout = nrgin - nr_inh
!
    count = 0
    DO
      count = count + 1
      ratrrb = ratrr
      alge = -(ratrr**(nrout)-1.0d0-rvdom*dr1inv*(ratrr-1.0d0))
      dalge = dble(nrout)*ratrr**(nrout-1) - rvdom*dr1inv
!test    alge = -((ratrr**nrout-1.0d0)/(ratrr-1.0d0)-rvdom*dr1inv)
!test    dalge = (dble(nrout-1)*ratrr**nrout-dble(nrout)*ratrr**(nrout-1)+1.0d0) &
!test    &     / (ratrr-1.0d0)**2
      ratrr = ratrr + alge/dalge
      error = 2.d0*(ratrr - ratrrb)/(ratrr + ratrrb)
! 
      IF (abs(error)<=1.d-15) EXIT 
      IF (count.ge.1000) exit
    END DO
!
    IF (count.ge.1000) cycle
! r-coordinate. interval dr
    drg(nr_inh+1) = dr1
    drginv(nr_inh+1) = 1.0d0/dr1
    do ir = nr_inh+2, nrgin
      drg(ir) = ratrr*drg(ir-1)
      drginv(ir) = 1.0d0/drg(ir)
    end do
!
    DO ir = nr_inh+1, nrgin
      rg(ir) = rg(ir-1) + drg(ir)
!      write(6,'(i5,1p,e23.15)') ir, rg(ir)
    end do
    if(dabs(rg(nrgin)-r_surf)>2.e-14 .and. itry==3)  stop "dabs(rg(nrgin)-r_surf)>2e-14"
 
    DO ir = nrgin+1, nrf
      rg(ir) = rg(nrgin) + dble(ir-nrgin)*drdr
!      write(6,'(i5,1p,e23.15)') ir, rg(ir)
    end do
!
    rdet  = (rg(nrf) - rgmid)
    rdetf = 0.0d0
    IF(dabs(rdet)< 2.e-14) then
      WRITE(6,*) 'good coordinate until rg(nrf)=', rg(nrf)
      WRITE(6,*) 'good coordinate itry =', itry
      WRITE(6,*) 'good coordinate GR ', rdet, ratrr
      WRITE(6,*) 'good coordinate FLUID ', rdetf, ratrf
      exit
    end if
    IF(dabs(rdet)>=2.e-14)then
      WRITE(6,*) ' bad coordinate until rg(nrf)=', rg(nrf)
      WRITE(6,*) ' bad coordinate itry =', itry
      WRITE(6,*) ' bad coordinate GR ', rdet, ratrr
      WRITE(6,*) ' bad coordinate FLUID ', rdetf, ratrf
      if (itry.eq.3) STOP
    END IF
  end do
!
! region III
!    drdr =  drg(nrgin)
!    drdrinv = 1.0d0/drdr
!    rvdom = 2.0d0*rgmid
!    nrout = nrf
!    ratrr = 2.0d0
!
!    DO
!      ratrrb = ratrr
!      alge = -(ratrr**(nrout+1)-ratrr-rvdom*drdrinv*(ratrr-1.0d0))
!      dalge = dble(nrout+1)*ratrr**nrout - 1.0d0 - rvdom*drdrinv
!      ratrr = ratrr + alge/dalge
!      error = 2.d0*(ratrr - ratrrb)/(ratrr + ratrrb)
 
!      IF (abs(error)<=1.d-15) EXIT 
!    END DO
! r-coordinate. interval dr
!    DO ir = nrgin+1, nrgin+nrf
!      drg(ir) = ratrr*drg(ir-1)
!      drginv(ir) = 1.0d0/drg(ir)
!    END DO
!
  do itry = 0, 3 
    if (itry.eq.0) ratrr = 1.05d-00 ! good for finer grid
    if (itry.eq.1) ratrr = 2.0d0   ! good for finer grid
    if (itry.eq.2) ratrr = 10.0d0  ! good for coarser grid
    if (itry.eq.3) ratrr = 50.0d0  ! good for coarsest grid
! 
    drdr =  drg(nrf)
    drdrinv = 1.0d0/drdr
    rvdom = rgout - rgmid
    nrout = nrg - nrf
!
    DO
      ratrrb = ratrr
!test    alge = -(ratrr**(nrout+1)-ratrr-rvdom*drdrinv*(ratrr-1.0d0))
!test    dalge = dble(nrout+1)*ratrr**nrout - 1.0d0 - rvdom*drdrinv
      alge = -(dble(nrout+1)*log(ratrr)-log(ratrr+rvdom*drdrinv*(ratrr-1.0d0)))
      dalge = dble(nrout+1)/ratrr - (1.0d0+rvdom*drdrinv)/ &
      &                             (ratrr+rvdom*drdrinv*(ratrr-1.0d0))
      ratrr = ratrr + alge/dalge
      error = 2.d0*(ratrr - ratrrb)/(ratrr + ratrrb)
      IF (abs(error)<=1.d-16) EXIT 
    END DO
!
!   r-coordinate. interval dr
    nr_count = nrf+1
    DO ir = nr_count, nrg
      drg(ir) = ratrr*drg(ir-1)
      drginv(ir) = 1.0d0/drg(ir)
    END DO
!   r-coordinate. grid ponit and mid point.
    rg(0) = rgin
    if (rg(0).le.1.0d-14) then 
      rginv(0) = 0.0d0
    else 
      rginv(0) = 1.0d0/rg(0)
    end if
    DO ir = 1, nrg
      rg(ir) = rg(ir-1) + drg(ir)
      hrg(ir) = (rg(ir) + rg(ir-1))*0.5d0
      rginv(ir) = 1.0d0/rg(ir)
      hrginv(ir) = 1.0d0/hrg(ir)
!      write(6,*)ir,rg(ir)
    END DO
!      WRITE(6,'(1p,2e12.4)') error, ratrr
!      WRITE(6,'(1p,2e12.4)') rgout, rg(nrg)
    rdet  = (rg(nrg) - rgout)/rgout
    rdetf = 0.0d0
    if (rgin.lt.1.0d0) rdetf =  rg(nrf) - rgmid
    IF(dabs(rdet)< tolerant_val) then
      WRITE(6,*) 'good coordinate until rg(nrg)=', rg(nrg)
      WRITE(6,*) 'good coordinate itry =', itry
      WRITE(6,*) 'good coordinate GR ', rdet, ratrr
      WRITE(6,*) 'good coordinate FLUID ', rdetf, ratrf
      exit
    end if
    IF(dabs(rdet)>=tolerant_val)then
      WRITE(6,*) ' bad coordinate until rg(nrg)=', rg(nrg)
      WRITE(6,*) ' bad coordinate itry =', itry
      WRITE(6,*) ' bad coordinate GR ', rdet, ratrr
      WRITE(6,*) ' bad coordinate FLUID ', rdetf, ratrf
      if (itry.eq.3) STOP
    END IF
  end do
! testtest
!do ir = 0, nrg
!write(6,*) rg(ir)
!end do
!stop
!
end subroutine grid_r_bht
!

end module coordinate_grav_r
