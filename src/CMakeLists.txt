# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(DataStructures)
add_subdirectory(Domain)
add_subdirectory(ErrorHandling)
add_subdirectory(Executables)
add_subdirectory(Informer)
add_subdirectory(IO)
add_subdirectory(Numerical)
add_subdirectory(Parallel)
add_subdirectory(Time)
add_subdirectory(Utilities)

# Propagate core libraries up to global scope.
set(SPECTRE_LIBRARIES
    "${SPECTRE_CORE_LIBRARIES};${SPECTRE_LIBRARIES}" PARENT_SCOPE)
