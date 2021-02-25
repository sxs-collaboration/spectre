# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Informer)

add_spectre_library(${LIBRARY})

# Note that this library has two special source files: InfoAtCompile.cpp and
# InfoAtLink.cpp. These files contain placeholder strings that CMake fills in
# (in copies of the files in the build directory) when configuring the build.
#
# We track the *generated* InfoAtCompile.cpp as a library source. This means
# that CMake must have been configured *before* checking library dependencies
# via the CMake function check_spectre_libs_dependencies.
#
# We do *not* track InfoAtLink.cpp as a library source, as it is used only to
# inject information late in the build process. The InfoAtLink target defined
# below determines the approriate flags for compiling InfoAtLink.cpp.
spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Informer.cpp
  Verbosity.cpp
  ${CMAKE_BINARY_DIR}/Informer/InfoAtCompile.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  InfoFromBuild.hpp
  Informer.hpp
  Tags.hpp
  Verbosity.hpp
  )

# The Darwin linker is not OK with undefined symbols at link time of the
# library, so we need to appease it. This is needed so compiling InfoAtLink.cpp
# can be deferred until we link an executable.
target_link_options(
  ${LIBRARY}
  PUBLIC "$<$<PLATFORM_ID:Darwin>:-Wl,-U,_info_from_build>")

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  Options
  PRIVATE
  ErrorHandling
  SystemUtilities
  )

# Define the `InfoAtLink` target to determine the appropriate flags for
# compiling `InfoAtLink.cpp`. The Python script stores the flags in the file
# `${CMAKE_BINARY_DIR}/Informer/InfoAtLink_flags.txt` so they can be used for
# compiling `InfoAtLink.cpp` at link time (see `WrapExecutableLinker.sh`).
configure_file(
  ${CMAKE_SOURCE_DIR}/src/Informer/InfoAtLinkExtractFlags.py
  ${CMAKE_BINARY_DIR}/Informer/InfoAtLinkExtractFlags.py
  )
set(INFO_AT_LINK_LIB InfoAtLink)
add_library(${INFO_AT_LINK_LIB} OBJECT InfoAtLink.cpp)
target_link_libraries(
  ${INFO_AT_LINK_LIB}
  PRIVATE
  Boost::boost
  )
set_target_properties(
  ${INFO_AT_LINK_LIB}
  PROPERTIES
  CXX_COMPILER_LAUNCHER
  ${CMAKE_BINARY_DIR}/Informer/InfoAtLinkExtractFlags.py
  )
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.15)
  set_target_properties(
    ${INFO_AT_LINK_LIB}
    PROPERTIES
    ADDITIONAL_CLEAN_FILES
    ${CMAKE_BINARY_DIR}/Informer/InfoAtLink_flags.txt
    )
endif()

add_subdirectory(Python)
