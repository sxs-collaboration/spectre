subroutine read_parameter_binary_excision_mpt(impt)
  use phys_constant, only : long
  use grid_parameter, only : nrf
  use coordinate_grav_r, only : rg
  use grid_parameter_binary_excision, only : ex_nrg, ex_ndis
  implicit none
  integer,intent(in)  :: impt
  character(len=1) :: np(5) = (/'1', '2','3', '4', '5'/)
  open(1,file='bin_ex_par_mpt'//np(impt)//'.dat',status='old')
  read(1,'(2i5)') ex_nrg, ex_ndis
  close(1)
!  if (nrf.ge.ex_nrg) stop ' nrf > ex_nrg '
  if (ex_nrg.ne.0) then 
    if (nrf.ge.ex_nrg) write(6,*) '** Warning ** nrf > ex_nrg '
    if (nrf.ge.ex_nrg) write(6,*) 'nrf = ', nrf, '  ex_nrg = ', ex_nrg
  end if
  if (ex_nrg.eq.0) write(6,*) 'No binary excision for patch =',impt 
end subroutine read_parameter_binary_excision_mpt

subroutine read_parameter_binary_excision_mpt_cactus(impt, dir_path)
  use phys_constant, only : long
  use grid_parameter, only : nrf
  use coordinate_grav_r, only : rg
  use grid_parameter_binary_excision, only : ex_nrg, ex_ndis
  implicit none
  integer,intent(in)  :: impt
  character*400, intent(in) :: dir_path
  character(len=1) :: np(5) = (/'1', '2','3', '4', '5'/)
  character*400 :: filepath
!  filepath="/home/fs01/ml2847/SPECTRE/BNS/ID/IRE3.0_SLy_SLy_010_M1.96/work_area_BNS//bin_ex_par_mpt1.dat"
  open(1,file=trim(dir_path)//'/'//'bin_ex_par_mpt'//np(impt)//'.dat',status='old')
!  open(1,file=trim(filepath),status='old')
  read(1,'(2i5)') ex_nrg, ex_ndis
  close(1)
!  if (nrf.ge.ex_nrg) stop ' nrf > ex_nrg '
  if (ex_nrg.ne.0) then 
    if (nrf.ge.ex_nrg) write(6,*) '** Warning ** nrf > ex_nrg '
    if (nrf.ge.ex_nrg) write(6,*) 'nrf = ', nrf, '  ex_nrg = ', ex_nrg
  end if
  if (ex_nrg.eq.0) write(6,*) 'No binary excision for patch =',impt 
end subroutine read_parameter_binary_excision_mpt_cactus
