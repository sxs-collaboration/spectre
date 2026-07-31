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
  )
  # libquadmath is not shipped on all targets (e.g. aarch64, where `long double`
  # is already IEEE-128-bit), so only link it when it was actually found.
  if(quadmath)
    target_link_libraries(${LIBRARY} PRIVATE ${quadmath})
  endif()
endif()

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  SpectreFlags
)
