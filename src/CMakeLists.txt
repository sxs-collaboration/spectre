# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(DataStructures)
add_subdirectory(Domain)
add_subdirectory(ErrorHandling)
add_subdirectory(Executables)
add_subdirectory(Numerical)

# Propagate core labraries up to global scope. GSL needs to be added explicitly here due
# to strange linking errors with GCC
set(SPECTRE_LIBRARIES
    "${SPECTRE_CORE_LIBRARIES};${SPECTRE_LIBRARIES};-lgsl -lgslcblas" PARENT_SCOPE)
