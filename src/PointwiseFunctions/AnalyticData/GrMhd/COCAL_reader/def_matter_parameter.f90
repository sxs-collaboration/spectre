module def_matter_parameter
  use phys_constant, only : long
  implicit none
  real(long)           ::  pinx, emdc
  real(long)           ::  ome, ber, radi
  character(len=2)     ::  ROT_LAW
  real(long)           ::  A2DR, DRAT_A2DR, index_DR, index_DRq
  real(long)           ::  B2DR, DRAT_B2DR, index_DRp
! ome  is the angular velocity \Omega, 
! ber  is the injection energy \epsilon, 
! radi is the radius of the star.
!
!  the following are used for spinning stars
  real(long)           ::  omespx, omespy, omespz, confpow
!  the following are used for eccentricity reduction
  real(long)           ::  velx, delome, delvel
end module def_matter_parameter
