# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Valencia)

set(SYSTEM_TESTS "${SYSTEM_TESTS};${RELATIVISTIC_EULER_TESTS}" PARENT_SCOPE)
set(LIBRARIES_TO_LINK "${LIBRARIES_TO_LINK}" PARENT_SCOPE)
