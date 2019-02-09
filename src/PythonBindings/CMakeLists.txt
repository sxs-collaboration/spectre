# Distributed under the MIT License.
# See LICENSE.txt for details.

# WARNING: This library should be linked as ${SPECTRE_LINK_PYBINDINGS}
#          for it to work properly. If you get errors such as CmiPrintf
#          is not defined, then you are not linking to it correctly.
set(LIBRARY "PyBindings")

set(LIBRARY_SOURCES
  CharmCompatibility.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC ${Boost_LIBRARIES}
  PUBLIC Utilities
  )
