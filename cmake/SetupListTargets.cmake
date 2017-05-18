# Distributed under the MIT License.
# See LICENSE.txt for details.

add_custom_target(
    list-targets
    COMMAND
    ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target help
    | grep -v " depend$"
    | grep -v " edit_cache$"
    | grep -v " module_"
    VERBATIM
)

add_custom_target(targets COMMAND DEPENDS list-targets)

add_custom_target(list COMMAND DEPENDS list-targets)
