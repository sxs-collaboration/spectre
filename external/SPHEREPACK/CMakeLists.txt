# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SPHEREPACK)

set(LIBRARY_SOURCES
    alf.f
    divgs.f
    gaqd.f
    gradgs.f
    hrfft.f
    shags.f
    shsgs.f
    slapgs.f
    sphcom.f
    vhags.f
    vhsgs.f
    vrtgs.f
    vtsgs.f
    )

# Starting in gcc8, gfortran warns about instances of Fortran77 coding style in
# certain SPHEREPACK files. The warning is:
#   Warning: Array reference at ... out of bounds ... in loop beginning ...
#
# Attempts to silence the warning by use of individual flags were unsuccessful,
# so instead we silence all warnings for the problematic files.
set(FILES_WITH_ARRAY_REFERENCES
    alf.f
    sphcom.f
    vhags.f
    vhsgs.f
    vtsgs.f
    )
set_source_files_properties(${FILES_WITH_ARRAY_REFERENCES}
    PROPERTIES COMPILE_FLAGS -w)

# Starting in gcc9, gfortran warns about deleted language features in certain
# SPHEREPACK files. The warnings are,
# * Warning: Fortran 2018 deleted feature: Arithmetic IF statement at ...
# * Warning: Fortran 2018 deleted feature: Shared DO termination label at ...
#
# We silence these warnings too.
# An alternative solution would be to specify the compiler flag `-std=legacy`.
set(FILES_WITH_DELETED_FEATURES
    gradgs.f
    hrfft.f
    shags.f
    shsgs.f
    )
set_source_files_properties(${FILES_WITH_DELETED_FEATURES}
    PROPERTIES COMPILE_FLAGS -w)

add_library(${LIBRARY} ${LIBRARY_SOURCES})

if(SPECTRE_Fortran_STATIC_LIBS)
  if (NOT $gfortran)
    find_library(gfortran NAMES libgfortran.a)
  endif()
  if (NOT $quadmath)
    find_library(quadmath NAMES libquadmath.a)
  endif()
  target_link_libraries(
    ${LIBRARY}
    PRIVATE
    ${gfortran}
    ${quadmath}
  )
endif()

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  SpectreFlags
)
