subroutine read_surf_parameter_mpt(impt)
  use phys_constant, only : long
  use grid_parameter
  use def_matter_parameter, only : omespx, omespy, omespz
  implicit none
  integer,intent(in)  :: impt
  character(len=1) :: np(5) = (/'1', '2','3', '4', '5'/)

  if(impt==1 .or. impt==2) then
    open(1,file='rnspar_surf_mpt'//np(impt)//'.dat',status='old')
    read(1,'(i5,4x,a1,3i5)') nrg_1, sw_eos, sw_sepa, sw_quant, sw_spin
!    read(1,'(1p,e14.6)') r_surf
    read(1,'(1p,1e23.15)') r_surf
    read(1,'(1p,1e23.15)') target_sepa
    read(1,'(1p,1e23.15)') target_qt
    read(1,'(1p,1e23.15)') target_sx
    read(1,'(1p,1e23.15)') target_sy
    read(1,'(1p,1e23.15)') target_sz
    read(1,'(1p,1e23.15)') omespx
    read(1,'(1p,1e23.15)') omespy
    read(1,'(1p,1e23.15)') omespz
    close(1)
!
    if (indata_type.eq.'3D') then
      open(2,file='r_surf_mpt'//np(impt)//'.dat',status='old')
      read(2,'(1p,1e23.15)') r_surf
      close(2)
    end if
    write(6,*) "Patch:",impt," r_surf=", r_surf
    write(6,*) "omespx,omespy,omespz=", omespx,omespy,omespz
  end if
end subroutine read_surf_parameter_mpt


subroutine read_surf_parameter_mpt_cactus(impt, dir_path)
  use phys_constant, only : long
  use grid_parameter
  use def_matter_parameter, only : omespx, omespy, omespz
  implicit none
  integer,intent(in)  :: impt
  character*400, intent(in) :: dir_path
  character*400 :: filepath
  character(len=1) :: np(5) = (/'1', '2','3', '4', '5'/)

  if(impt==1 .or. impt==2) then
!    print *, "In read_surf_parameter_mpt_cactus, dir_path before called is:" ,dir_path
!    filepath= "/home/fs01/ml2847/SPECTRE/BNS/ID/IRE3.0_SLy_SLy_010_M1.96/work_area_BNS/rnspar_surf_mpt1.dat"
    open(1,file=trim(dir_path)//'/'//'rnspar_surf_mpt'//np(impt)//'.dat',status='old')
!    open(1,file=trim(filepath),status='old')
    read(1,'(i5,4x,a1,3i5)') nrg_1, sw_eos, sw_sepa, sw_quant, sw_spin
!    read(1,'(1p,e14.6)') r_surf
    read(1,'(1p,1e23.15)') r_surf
    read(1,'(1p,1e23.15)') target_sepa
    read(1,'(1p,1e23.15)') target_qt
    read(1,'(1p,1e23.15)') target_sx
    read(1,'(1p,1e23.15)') target_sy
    read(1,'(1p,1e23.15)') target_sz
    read(1,'(1p,1e23.15)') omespx
    read(1,'(1p,1e23.15)') omespy
    read(1,'(1p,1e23.15)') omespz
    close(1)
!
    if (indata_type.eq.'3D') then
!      filepath= "/home/fs01/ml2847/SPECTRE/BNS/ID/IRE3.0_SLy_SLy_010_M1.96/work_area_BNS/r_surf_mpt1.dat"
      open(2,file=trim(dir_path)//'/'//'r_surf_mpt'//np(impt)//'.dat',status='old')
!      open(2,file=trim(filepath),status='old')
      read(2,'(1p,1e23.15)') r_surf
      close(2)
    end if
    write(6,*) "Patch:",impt," r_surf=", r_surf
    write(6,*) "omespx,omespy,omespz=", omespx,omespy,omespz
  end if
end subroutine read_surf_parameter_mpt_cactus

