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
# inject information late in the build process. But note that InfoAtLink.cpp
# depends on Boost.
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

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  Options
  PRIVATE
  ErrorHandling
  )

add_subdirectory(Python)
