# Distributed under the MIT License.
# See LICENSE.txt for details.

if (${BUILD_PYTHON_BINDINGS})
  set(LIBRARY "PyBindings")

  set(LIBRARY_SOURCES
    CharmCompatibility.cpp
    )

  add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})
endif()
