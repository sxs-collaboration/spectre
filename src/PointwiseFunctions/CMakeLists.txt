# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY PointwiseFunctions)

set(LIBRARY_SOURCES)

add_subdirectory(MathFunctions)

add_library(${LIBRARY} ${LIBRARY_SOURCES})

# ${LIBRARY} is considered a core library on which all executables depend
set(SPECTRE_CORE_LIBRARIES "${SPECTRE_CORE_LIBRARIES};${LIBRARY}" PARENT_SCOPE)
