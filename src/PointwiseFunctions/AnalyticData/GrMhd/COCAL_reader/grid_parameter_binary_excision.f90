module grid_parameter_binary_excision
  use phys_constant, only : long
  implicit none
  integer :: ex_nrg, ex_ndis ! size of excision radius
  real(long) :: ex_radius, ex_rgin, ex_rgmid, ex_rgout
contains
subroutine read_parameter_binary_excision
  use grid_parameter, only : nrf
  implicit none
  open(1,file='bin_ex_par.dat',status='old')
  read(1,'(2i5)') ex_nrg, ex_ndis
  close(1)
!  if (nrf.ge.ex_nrg) stop ' nrf > ex_nrg '
  if (nrf.ge.ex_nrg) write(6,*) '** Warning ** nrf > ex_nrg '
  if (nrf.ge.ex_nrg) write(6,*) 'nrf = ', nrf, '  ex_nrg = ', ex_nrg
  if (ex_nrg.eq.0)   write(6,*) ' No binary excision '
end subroutine read_parameter_binary_excision
!
subroutine calc_parameter_binary_excision
  use grid_parameter, only : nrf, nrg
  use coordinate_grav_r, only : rg, drg
  use coordinate_grav_theta
  use coordinate_grav_phi
  use def_binary_parameter, only : sepa, dis
  implicit none
  integer :: irg
  sepa = 2.0d0*rg(ex_nrg + ex_ndis)
  dis  =       rg(ex_nrg + ex_ndis)
  ex_radius = rg(ex_nrg)
  ex_rgin   = dis + dis - rg(ex_nrg)
  ex_rgmid  = sepa
  ex_rgout  = sepa + rg(ex_nrg)
  if (ex_nrg.eq.0) then 
    sepa = 0.0d0
    dis  = 0.0d0
    ex_radius = 0.0d0
    ex_rgin   = 0.0d0
    ex_rgmid  = 0.0d0
    ex_rgout  = 0.0d0
  end if
!
end subroutine calc_parameter_binary_excision
!
subroutine IO_printout_grid_data
  use grid_parameter, only : nrf, nrg
  use coordinate_grav_r, only : rg, drg
  use coordinate_grav_theta
  use coordinate_grav_phi
  use def_binary_parameter, only : sepa, dis
  implicit none
  integer :: irg
  open(1,file='grid_data.dat',status='unknown')
!  write(1,'(a5,1p,e20.12)') 'sepa=', sepa
!  write(1,'(a5,1p,e20.12)') 'dis =', dis
!  write(1,'(a8,i3,a8,1p,e20.12,a11)') 'ex_nrg =', ex_nrg, '     rg=', rg(ex_nrg) , ' =ex_radius'
!  write(1,'(a9,i3)') 'ex_ndis =', ex_ndis
!  write(1,'(a9,1p,e20.12)') 'ex_rgin =', ex_rgin
!  write(1,'(a9,1p,e20.12)') 'ex_rgmid=', ex_rgmid
!  write(1,'(a9,1p,e20.12)') 'ex_rgout=', ex_rgout
  write(1,'(a4,i3,a10,1p,e20.12)') 'ntg=', ntg, '     dthg=', dthg
  write(1,'(a4,i3,a10,1p,e20.12)') 'npg=', npg, '     dphg=', dphig
  write(1,'(a100)') '           ex_radius<                dis<            ex_rgin<               sepa<           ex_rgout'
  write(1,'(1p,6e20.12)') ex_radius,dis,ex_rgin,sepa,ex_rgout
!
  write(1,'(a37,i3,a2,1p,e20.12)') '..................................rg(',0,')=', rg(0)
!
  do irg=1,ex_nrg-1
    write(1,'(a4,i3,a2,1p,e20.12,a5,a3,i3,a2,1p,e20.12)') 'drg(',irg,')=',drg(irg),'     ','rg(',irg,')=',rg(irg)
  end do

  write(1,'(a4,i3,a2,1p,e20.12,a5,a3,i3,a2,1p,e20.12,a24)') 'drg(',ex_nrg,')=',drg(ex_nrg),'     ','rg(',irg,')=',&
      &  rg(ex_nrg), '    ex_radius=rg(ex_nrg)' 
!
  do irg=ex_nrg+1,(ex_nrg+ex_ndis-1)
    write(1,'(a4,i3,a2,1p,e20.12,a5,a3,i3,a2,1p,e20.12)') 'drg(',irg,')=',drg(irg),'     ','rg(',irg,')=',rg(irg)
  end do
!
  irg=ex_nrg+ex_ndis
!
  write(1,'(a4,i3,a2,1p,e20.12,a5,a3,i3,a2,1p,e20.12,a24)') 'drg(',irg,')=',drg(irg),'     ','rg(',irg,')=',rg(irg) &
    &   , '  rg(ex_nrg+ex_ndis)=dis'
!
  do irg=(ex_nrg+ex_ndis+1),nrg
    write(1,'(a4,i3,a2,1p,e20.12,a5,a3,i3,a2,1p,e20.12)') 'drg(',irg,')=',drg(irg),'     ','rg(',irg,')=',rg(irg)
  end do   
!
  close(1)
end subroutine IO_printout_grid_data
!
subroutine IO_printout_grid_data_mpt(impt)
  use grid_parameter
  use coordinate_grav_r
  use coordinate_grav_theta
  use coordinate_grav_phi
  use def_binary_parameter
  implicit none
  integer :: irg, impt
  character(len=1) :: np(5) = (/'1', '2', '3', '4', '5'/)

  open(1,file='grid_data_mpt'//np(impt)//'.dat',status='unknown')
  if(impt==1 .or. impt==2) then
  write(1,'(a4,i3,a10,1p,e20.12)') 'ntg=', ntg, '     dthg=', dthg
  write(1,'(a4,i3,a10,1p,e20.12)') 'npg=', npg, '     dphg=', dphig
  write(1,'(a100)') '           ex_radius<                dis<            ex_rgin<               sepa<           ex_rgout'
  write(1,'(1p,6e20.12)') ex_radius,dis,ex_rgin,sepa,ex_rgout
  write(1,'(a35,e20.12)') " sin(half angle of excised sphere)=", ex_radius/sepa
  write(1,'(a35,2p,e20.12,a9)') "...or half angle of excised sphere=", DASIN(ex_radius/sepa)*180.0d0/pi, "  degrees"
  write(1,*) " "
  write(1,*) " "
  write(1,*) 'REGION S  ----------------------------------------------------------------'
  write(1,'(a41,1p,e20.12)') 'nrf         interval : drg(nrf  )      =', drg(nrf)
  write(1,*) " "
  write(1,*) 'REGION  I ----------------------------------------------------------------'
  write(1,'(a41,1p,e20.12)') 'nrf+1       interval : drg(nrf+1)      =', drg(nrf+1)
  write(1,'(a41,1p,e20.12)') 'nrf+2       interval : drg(nrf+2)      =', drg(nrf+2)
  write(1,*) '........................................................'
  write(1,'(a41,1p,e20.12)') 'nrg_1-1     interval : drg(nrg_1-1)    =', drg(nrg_1-1)
  write(1,'(a41,1p,e20.12)') 'nrg_1       interval : drg(nrg_1  )    =', drg(nrg_1)
  write(1,*) " "
  write(1,*) 'REGION  II----------------------------------------------------------------'
  write(1,'(a41,1p,e20.12)') 'nrg_1+1     interval : drg(nrg_1+1)    =', drg(nrg_1+1)
  write(1,*) '........................................................'
  write(1,'(a41,1p,e20.12)') 'nrgin       interval : drg(nrgin)      =', drg(nrgin)
  write(1,*) 'REGION III----------------------------------------------------------------'
  write(1,'(a41,1p,e20.12)') 'nrgin+1     interval : drg(nrgin+1)    =', drg(nrgin+1)
  write(1,'(a41,1p,e20.12)') 'nrgin+2     interval : drg(nrgin+2)    =', drg(nrgin+2)
  write(1,*) '........................................................'
  write(1,'(a41,1p,e20.12)') 'nrgin+nrf-1 interval : drg(nrgin+nrf-1)=', drg(nrgin+nrf-1)
  write(1,'(a41,1p,e20.12)') 'nrgin+nrf   interval : drg(nrgin+nrf  )=', drg(nrgin+nrf)
  write(1,*) " "
  write(1,*) 'REGION IV ----------------------------------------------------------------'
  write(1,'(a41,1p,e20.12)') 'nrgin+nrf+1 interval : drg(nrgin+nrf+1)=', drg(nrgin+nrf+1)
  write(1,'(a41,1p,e20.12)') 'nrgin+nrf+2 interval : drg(nrgin+nrf+2)=', drg(nrgin+nrf+2)
  write(1,*) '........................................................'
  write(1,'(a41,1p,e20.12)') 'nrg-1       interval : drg(nrg-1)      =', drg(nrg-1)
  write(1,'(a41,1p,e20.12)') 'nrg         interval : drg(nrg)        =', drg(nrg)
  write(1,*) "--------------------------------------------------------------------------"
!
  write(1,'(a37,i3,a2,1p,e20.12)') '..................................rg(',0,')=', rg(0)
!
  do irg=1,ex_nrg-1
    write(1,'(a4,i3,a2,1p,e20.12,a5,a3,i3,a2,1p,e20.12)') 'drg(',irg,')=',drg(irg),'     ','rg(',irg,')=',rg(irg)
  end do

  write(1,'(a4,i3,a2,1p,e20.12,a5,a3,i3,a2,1p,e20.12,a24)') 'drg(',ex_nrg,')=',drg(ex_nrg),'     ','rg(',irg,')=',&
      &  rg(ex_nrg), '    ex_radius=rg(ex_nrg)' 
!
  do irg=ex_nrg+1,(ex_nrg+ex_ndis-1)
    write(1,'(a4,i3,a2,1p,e20.12,a5,a3,i3,a2,1p,e20.12)') 'drg(',irg,')=',drg(irg),'     ','rg(',irg,')=',rg(irg)
  end do
!
  irg=ex_nrg+ex_ndis
!
  write(1,'(a4,i3,a2,1p,e20.12,a5,a3,i3,a2,1p,e20.12,a24)') 'drg(',irg,')=',drg(irg),'     ','rg(',irg,')=',rg(irg) &
    &   , '  rg(ex_nrg+ex_ndis)=dis'
!
  do irg=(ex_nrg+ex_ndis+1),nrg
    write(1,'(a4,i3,a2,1p,e20.12,a5,a3,i3,a2,1p,e20.12)') 'drg(',irg,')=',drg(irg),'     ','rg(',irg,')=',rg(irg)
  end do 
  else
  write(1,'(a4,i3,a10,1p,e20.12)') 'ntg=', ntg, '     dthg=', dthg
  write(1,'(a4,i3,a10,1p,e20.12)') 'npg=', npg, '     dphg=', dphig
  write(1,'(a100)') '           ex_radius<                dis<            ex_rgin<               sepa<           ex_rgout'
  write(1,'(1p,6e20.12)') ex_radius,dis,ex_rgin,sepa,ex_rgout
!
  write(1,'(i5,1p,3e20.12)') 0, rg(0)
  do irg=1,nrg
    write(1,'(i5,1p,3e20.12)') irg, rg(irg), drg(irg), drg(irg)*drginv(irg)
  end do
  end if  
  close(1)
 
end subroutine IO_printout_grid_data_mpt

end module grid_parameter_binary_excision
