# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "CceAnalyticSolutionsHelpers")

set(LIBRARY_SOURCES
  AnalyticDataHelpers.cpp
  )

add_spectre_library(${LIBRARY} ${SPECTRE_TEST_LIBS_TYPE} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Cce
  DataStructures
  Framework
  Utilities
  )
