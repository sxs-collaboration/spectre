# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Actions)
add_subdirectory(DiscontinuousGalerkin)
add_subdirectory(EventsAndTriggers)
add_subdirectory(Systems)

set(SPECTRE_TESTS "${SPECTRE_TESTS};${EVOLUTION_TESTS}" PARENT_SCOPE)
set(LIBRARIES_TO_LINK "${LIBRARIES_TO_LINK}" PARENT_SCOPE)
