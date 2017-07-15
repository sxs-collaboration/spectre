# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE ParallelInfo)

set(EXECUTABLE_SOURCES
    ParallelInfo.cpp
    ${EXECUTABLE}.decl.h
)

add_charm_mainmodule(${EXECUTABLE})

add_executable(
    ${EXECUTABLE}
    ${EXECUTABLE_SOURCES}
)

target_link_libraries(
    ${EXECUTABLE}
    ErrorHandling
)
