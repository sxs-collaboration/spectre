# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DomainCreators)

set(LIBRARY_SOURCES
  Brick.cpp
  Disk.cpp
  Interval.cpp
  Rectangle.cpp
  RegisterDerivedWithCharm.cpp
  Shell.cpp
  Sphere.cpp
  )

add_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE Domain
  INTERFACE ErrorHandling
  )
