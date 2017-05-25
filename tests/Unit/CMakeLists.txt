# Distributed under the MIT License.
# See LICENSE.txt for details.

set(executable RunTests)

set(SPECTRE_TESTS "")

add_subdirectory(Utilities)

add_charm_mainmodule(${executable})

add_executable(
        ${executable}
        ${executable}.cpp
        ${executable}.decl.h
        ${executable}.def.h
        ${SPECTRE_TESTS}
)

target_link_libraries(
        ${executable}
)

spectre_add_catch_tests(${executable})

