# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Catch2 3.4.0)

if (NOT Catch2_FOUND)
  if (NOT SPECTRE_FETCH_MISSING_DEPS)
    message(FATAL_ERROR "Could not find Catch2. If you want to fetch "
      "missing dependencies automatically, set SPECTRE_FETCH_MISSING_DEPS=ON.")
  endif()
  message(STATUS "Fetching Catch2")
  include(FetchContent)
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
    GIT_SHALLOW TRUE
    ${SPECTRE_FETCHCONTENT_BASE_ARGS}
  )
  FetchContent_MakeAvailable(Catch2)
endif()
