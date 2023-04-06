# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Framework)

add_subdirectory(Tests)

set(LIBRARY_SOURCES
  SetupLocalPythonEnvironment.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Boost::boost
  DataStructuresHelpers
  Options
  Parallel
  SystemUtilities

  PRIVATE
  Informer

  PUBLIC
  Charmxx::charmxx
  DataStructures
  ErrorHandling
  Python::NumPy
  Python::Python
  Serialization
  Utilities
  )

# Give SetupLocalPythonEnvironment access to installed Python packages.
# The Python interpreter usually takes care of adding this directory to the path
# at launch, but in SetupLocalPythonEnvironment we just link in Python. The
# linked Python libs may reside in a different place than the Python interpreter
# (in particular when the Python interpreter was symlinked to a venv). Thefore,
# SetupLocalPythonEnvironment needs to add this directory to the path as well.
target_compile_definitions(
  ${LIBRARY}
  PRIVATE
  PYTHON_SITELIB="${Python_SITELIB}"
)
