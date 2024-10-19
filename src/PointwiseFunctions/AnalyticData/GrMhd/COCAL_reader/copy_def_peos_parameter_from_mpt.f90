subroutine copy_def_peos_parameter_from_mpt(impt)
  use def_peos_parameter      !abc,abi,rhoi,qi,hi,nphase,rhoini_cgs,emdini_gcm1
  use def_peos_parameter_mpt
  use copy_array_static_2dto1d_mpt
  implicit none
  integer :: i, impt
!  
  call copy_arraystatic_2dto1d_mpt(impt, abc_   , abc,    0, nnpeos)
  call copy_arraystatic_2dto1d_mpt(impt, abi_   , abi,    0, nnpeos)
  call copy_arraystatic_2dto1d_mpt(impt, rhoi_  , rhoi,   0, nnpeos)
  call copy_arraystatic_2dto1d_mpt(impt, qi_    , qi,     0, nnpeos)
  call copy_arraystatic_2dto1d_mpt(impt, hi_    , hi,     0, nnpeos)
  call copy_arraystatic_2dto1d_mpt(impt, abccgs_, abccgs, 0, nnpeos)
  call copy_arraystatic_2dto1d_mpt(impt, rhocgs_, rhocgs, 0, nnpeos)
!
  i=0
  i=i+1; rhoini_cgs  = def_peos_param_real_(i,impt)
  i=i+1; rhoini_gcm1 = def_peos_param_real_(i,impt) 
  i=i+1; emdini_gcm1 = def_peos_param_real_(i,impt) 
  i=i+1; sgma        = def_peos_param_real_(i,impt) 
  i=i+1; constqc     = def_peos_param_real_(i,impt) 
  i=i+1; cbar        = def_peos_param_real_(i,impt) 
!
  i=0
  i=i+1; nphase = def_peos_param_int_(i,impt)
!
end subroutine copy_def_peos_parameter_from_mpt
