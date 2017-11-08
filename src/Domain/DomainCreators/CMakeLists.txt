# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DomainCreators)

set(LIBRARY_SOURCES
  Brick.cpp
  Interval.cpp
  Rectangle.cpp
  RegisterDerivedWithCharm.cpp
  )

add_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE Domain
  INTERFACE ErrorHandling
  )
