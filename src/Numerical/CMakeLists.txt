# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY NumericalAlgorithms)

set(LIBRARY_SOURCES)

add_subdirectory(LinearAlgebra)
add_subdirectory(RootFinding)
add_subdirectory(Spectral)

add_library(${LIBRARY} ${LIBRARY_SOURCES})

# ${LIBRARY} is considered a core library on which all executables depend
set(SPECTRE_CORE_LIBRARIES "${SPECTRE_CORE_LIBRARIES};${LIBRARY}" PARENT_SCOPE)
