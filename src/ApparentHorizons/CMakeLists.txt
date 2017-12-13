# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ApparentHorizons)

set(LIBRARY_SOURCES
    SpherepackIterator.cpp
    Strahlkorper.cpp
    StrahlkorperDataBox.cpp
    YlmSpherepack.cpp
    YlmSpherepackHelper.cpp
    )

add_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE ErrorHandling
  INTERFACE SPHEREPACK
  )
