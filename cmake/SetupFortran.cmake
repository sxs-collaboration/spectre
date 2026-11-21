# Distributed under the MIT License.
# See LICENSE.txt for details.

option(SPECTRE_Fortran_STATIC_LIBS
  "Link static versions of libgfortran and libquadmath" OFF)

if(SPECTRE_Fortran_STATIC_LIBS)
  unset(CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES)
  find_library(gfortran NAMES libgfortran.a
    HINTS ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES})
  find_library(quadmath NAMES libquadmath.a
    HINTS ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES})
endif()
