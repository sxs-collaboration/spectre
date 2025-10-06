# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY slatec)

set(LIBRARY_SOURCES
    drc3jj.f
    )

add_library(${LIBRARY} ${LIBRARY_SOURCES})
target_compile_options(${LIBRARY} PRIVATE -std=legacy)

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
