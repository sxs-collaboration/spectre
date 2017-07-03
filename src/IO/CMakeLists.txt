# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY IO)

set(LIBRARY_SOURCES
    H5/Dat.cpp
    H5/File.cpp
    H5/Header.cpp
    H5/Helpers.cpp
    H5/OpenGroup.cpp
    H5/Version.cpp
)

add_library(${LIBRARY} ${LIBRARY_SOURCES})

# ${LIBRARY} is considered a core library on which all executables depend
set(SPECTRE_CORE_LIBRARIES "${SPECTRE_CORE_LIBRARIES};${LIBRARY}" PARENT_SCOPE)
