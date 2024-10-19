subroutine peos_initialize_mpt(impt)
!
  use phys_constant        !g,c,solmas,nnpeos
  use def_peos_parameter   !abc,abi,rhoi,qi,hi,nphase,rhoini_cgs,emdini_gcm1
  implicit none
  integer,intent(in)  :: impt
  character(len=1) :: np(2) = (/'1', '2'/)
!
  real(8) :: rho_0, pre_0, facrho, facpre, fac2, gg, cc, ss
  integer :: ii, iphase
!
  open(850,file='peos_parameter_mpt'//np(impt)//'.dat',status='old')
  read(850,'(8x,1i5,es13.5)') nphase, rhoini_cgs
  read(850,'(2es13.5)') rho_0, pre_0
  do ii = nphase, 0, -1
    read(850,'(2es13.5)') rhocgs(ii), abi(ii)
  end do
  close(850)
!
! --  cgs to g = c = msol = 1 unit.
! --  assume pre = pre_0 dyn/cm^2 at rho = rho_0 gr/cm^3.
! --  typically pre_0 = 1.0d+37 dyn/cm^2 
! --  and       rho_0 = 1.0d+16  gr/cm^3.
! --  rescale interface values
!
  facrho = (g/c**2)**3*solmas**2
  facpre = g**3*solmas**2/c**8
!
  do ii = 0, nphase
    rhoi(ii) = facrho*rhocgs(ii)
  end do
!
  call peos_lookup(rho_0,rhocgs,iphase)
!    
  abc(iphase) = pre_0/rho_0**abi(iphase)
  abc(iphase) = facpre/facrho**abi(iphase)*abc(iphase)
  abccgs(iphase) = pre_0/(rho_0**abi(iphase))
!
  if (iphase.gt.0) then
    do ii = iphase-1, 0, -1
      abc(   ii) = rhoi(  ii)**(abi(ii+1)-abi(ii))*abc(   ii+1)
      abccgs(ii) = rhocgs(ii)**(abi(ii+1)-abi(ii))*abccgs(ii+1)
    end do
  end if
  if (iphase.lt.nphase) then
    do ii = iphase+1, nphase
      abc(   ii) = rhoi(  ii-1)**(abi(ii-1)-abi(ii))*abc(   ii-1)
      abccgs(ii) = rhocgs(ii-1)**(abi(ii-1)-abi(ii))*abccgs(ii-1)
    end do
  end if
!
  do ii = 0, nphase
    qi(ii) = abc(ii)*rhoi(ii)**(abi(ii)-1.0d0)
  end do
!
  hi(0) = 1.0d0
  do ii = 1, nphase
    fac2 = abi(ii)/(abi(ii) - 1.0d0)
    hi(ii) = hi(ii-1) + fac2*(qi(ii) - qi(ii-1))
  end do
!
  open(860,file='peos_parameter_output_mpt'//np(impt)//'.dat',status='unknown')
  write(860,'(a1,8x,i5)')'#', nphase
  do ii = 0, nphase
    write(860,'(i5,10es13.5)') ii, abc(ii), abi(ii), rhoi(ii), &
    &                           qi(ii), hi(ii), abccgs(ii), rhocgs(ii), &
    &                           abccgs(ii)*rhocgs(ii)**abi(ii)
  end do
  close(860)
!
  rhoini_gcm1 = facrho*rhoini_cgs
  call peos_lookup(rhoini_gcm1,rhoi,iphase)
  emdini_gcm1 = abc(iphase)*rhoini_gcm1**(abi(iphase)-1.0d0)
!
end subroutine peos_initialize_mpt


subroutine peos_initialize_mpt_cactus(impt, dir_path)
!
  use phys_constant        !g,c,solmas,nnpeos
  use def_peos_parameter   !abc,abi,rhoi,qi,hi,nphase,rhoini_cgs,emdini_gcm1
  implicit none
  integer,intent(in)  :: impt
  character*400, intent(in) :: dir_path
  character(len=1) :: np(2) = (/'1', '2'/)
  character*400 :: filepath
!
  real(8) :: rho_0, pre_0, facrho, facpre, fac2, gg, cc, ss
  integer :: ii, iphase
!
!  filepath = "/home/fs01/ml2847/SPECTRE/BNS/ID/IRE3.0_SLy_SLy_010_M1.96/work_area_BNS/peos_parameter_mpt1.dat"
  open(850,file=trim(dir_path)//'/'//'peos_parameter_mpt'//np(impt)//'.dat',status='old')
!  open(850,file=trim(filepath),status='old')
  read(850,'(8x,1i5,es13.5)') nphase, rhoini_cgs
  read(850,'(2es13.5)') rho_0, pre_0
  do ii = nphase, 0, -1
    read(850,'(2es13.5)') rhocgs(ii), abi(ii)
  end do
  close(850)
!
! --  cgs to g = c = msol = 1 unit.
! --  assume pre = pre_0 dyn/cm^2 at rho = rho_0 gr/cm^3.
! --  typically pre_0 = 1.0d+37 dyn/cm^2 
! --  and       rho_0 = 1.0d+16  gr/cm^3.
! --  rescale interface values
!
  facrho = (g/c**2)**3*solmas**2
  facpre = g**3*solmas**2/c**8
!
  do ii = 0, nphase
    rhoi(ii) = facrho*rhocgs(ii)
  end do
!
  call peos_lookup(rho_0,rhocgs,iphase)
!    
  abc(iphase) = pre_0/rho_0**abi(iphase)
  abc(iphase) = facpre/facrho**abi(iphase)*abc(iphase)
  abccgs(iphase) = pre_0/(rho_0**abi(iphase))
!
  if (iphase.gt.0) then
    do ii = iphase-1, 0, -1
      abc(   ii) = rhoi(  ii)**(abi(ii+1)-abi(ii))*abc(   ii+1)
      abccgs(ii) = rhocgs(ii)**(abi(ii+1)-abi(ii))*abccgs(ii+1)
    end do
  end if
  if (iphase.lt.nphase) then
    do ii = iphase+1, nphase
      abc(   ii) = rhoi(  ii-1)**(abi(ii-1)-abi(ii))*abc(   ii-1)
      abccgs(ii) = rhocgs(ii-1)**(abi(ii-1)-abi(ii))*abccgs(ii-1)
    end do
  end if
!
  do ii = 0, nphase
    qi(ii) = abc(ii)*rhoi(ii)**(abi(ii)-1.0d0)
  end do
!
  hi(0) = 1.0d0
  do ii = 1, nphase
    fac2 = abi(ii)/(abi(ii) - 1.0d0)
    hi(ii) = hi(ii-1) + fac2*(qi(ii) - qi(ii-1))
  end do
!
!  open(860,file='peos_parameter_output_mpt'//np(impt)//'.dat',status='unknown')
!  write(860,'(a1,8x,i5)')'#', nphase
!  do ii = 0, nphase
!    write(860,'(i5,10es13.5)') ii, abc(ii), abi(ii), rhoi(ii), &
!    &                           qi(ii), hi(ii), abccgs(ii), rhocgs(ii), &
!    &                           abccgs(ii)*rhocgs(ii)**abi(ii)
!  end do
!  close(860)
!
  rhoini_gcm1 = facrho*rhoini_cgs
  call peos_lookup(rhoini_gcm1,rhoi,iphase)
  emdini_gcm1 = abc(iphase)*rhoini_gcm1**(abi(iphase)-1.0d0)
!
end subroutine peos_initialize_mpt_cactus
