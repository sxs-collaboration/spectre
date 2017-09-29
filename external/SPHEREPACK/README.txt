This is a modified version of SPHEREPACK 3.2.
The SPHEREPACK 3.2 license can be found in LICENSE.txt.

Only 13 FORTRAN files, with the functionality we need for SpECTRE,
have been retained here.  The unmodified SPHEREPACK contains
additional files with more functionality than we need, and we do not
include these files, nor do we include the SPHEREPACK makefiles or
tests.

A summary of the modifications:

1) We have added our own Spherepack.hpp for #including SPHEREPACK
   subroutines into C/C++.
2) All computations were changed to double precision.
3) divgs, gradgs, shags, shsgs, slapgs, vhags, vhsgs, vrtgs, vtsgs now
   have the ability to loop over 'offsets' and do multiple
   computations at once with a single call to a SPHEREPACK function.
   This is useful if you have a (r,theta,phi) grid, and you need to do
   spherical transforms on (theta,phi) for multiple values of r.
4) Bugfix in vtsgs.f, where mmax was computed incorrectly if nlon is even.
5) shsgs and shags no longer have limitations on the size of the passed-in
   coefficient arrays 'a' and 'b'.  They also take additional arguments,
   'mdabmax, 'ndabmax'.  Coefficients that do not fit into the provided
   storage are ignored / treated as zero / not computed.   The same holds
   for coefficients with indices above {m,n}dabmax.  This allows for
   more efficient implementation of definite integrals, where now
   one can request computation of the l=0 coefficient only.   This also
   helps in changing resolution, since restriction or prolongation
   of the coefficient array is no longer (explicitly) needed.
6) Given that we so far do not need to call SPHEREPACK with symmetries
   (e.g. equatorial symmetry, octant symmetry, etc) enforced, the above
   changes were not tested with SPHEREPACK's symmetry options turned on.
   In order to avoid surprises, SPHEREPACK's symmetry options are disabled.
   If you need symmetries, please test them.
