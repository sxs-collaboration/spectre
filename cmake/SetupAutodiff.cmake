# Distributed under the MIT License.
# See LICENSE.txt for details.

# Find autodiff (https://github.com/autodiff/autodiff)
#
# Note that Boost also has an autodiff module, but it uses templates to
# distinguish multiple variables. This means a std::array or Tensor can't hold
# the Boost autodiff variables. For example, to evaluate a 2D function f(x, y),
# the type for x would be fvar<double> and the type for y would be the nested
# fvar<fvar<double>>. On the other hand, the autodiff library works with a
# simpler autodiff::dual or autodiff::var type, which can be stored in a
# std::array or Tensor.

option(SPECTRE_AUTODIFF "Enable automatic differentiation" OFF)

if (NOT SPECTRE_AUTODIFF)
  return()
endif()

# we assume the found autodiff is newer than the GIT_TAG below,
# which has not been in autodiff's official release.
find_package(autodiff QUIET)

if (NOT autodiff_FOUND)
  if (NOT SPECTRE_FETCH_MISSING_DEPS)
    message(FATAL_ERROR "Could not find autodiff. If you want to fetch "
      "missing dependencies automatically, set SPECTRE_FETCH_MISSING_DEPS=ON.")
  endif()

  message(STATUS "Fetching autodiff")
  include(FetchContent)
  FetchContent_Declare(autodiff
      GIT_REPOSITORY https://github.com/autodiff/autodiff
      # Choose an unreleased version on top of v1.1.2 that makes dependence on
      # Eigen optional.
      GIT_TAG cc2aa5726fdbb258d097f87b97da3d1022f8394e
      ${SPECTRE_FETCHCONTENT_BASE_ARGS}
  )
  set(AUTODIFF_BUILD_TESTS OFF CACHE BOOL "Build autodiff tests")
  set(AUTODIFF_BUILD_EXAMPLES OFF CACHE BOOL "Build autodiff examples")
  set(AUTODIFF_BUILD_PYTHON OFF CACHE BOOL "Build autodiff Python bindings")
  set(AUTODIFF_BUILD_DOCS OFF CACHE BOOL "Build autodiff documentation")
  FetchContent_MakeAvailable(autodiff)

  if (CMAKE_VERSION VERSION_LESS 3.25)
    get_target_property(AUTODIFF_IID autodiff INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(autodiff PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${AUTODIFF_IID}")
  endif()
endif()

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  autodiff::autodiff
  )
