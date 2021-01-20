# Distributed under the MIT License.
# See LICENSE.txt for details.

option(
  USE_PCH
  "Use precompiled header file tools/SpectrePch.hpp"
  ON
  )

# Instructions for using the PCH:
#
# Targets can use the PCH generated for the `SpectrePch` library. They must also
# link the `SpectrePchFlags` library so they compile with the same flags as the
# PCH. Make sure to check the PCH exist before using them:
#
#     if(TARGET SpectrePch)
#       target_precompile_headers(${TARGET_NAME} REUSE_FROM SpectrePch)
#       target_link_libraries(${TARGET_NAME} PRIVATE SpectrePchFlags)
#     endif()

if(USE_PCH AND CMAKE_VERSION VERSION_LESS "3.16.0")
  message(WARNING "PCH not supported for CMake versions below 3.16, building "
    "without PCH. Consider upgrading your CMake version to use the PCH.")
  set(USE_PCH OFF)
endif()

if (USE_PCH)
  # This library tracks all compiler flags to build the PCH. Targets that use
  # the PCH must link this library so they compile with the same flags as the
  # PCH.
  add_library(SpectrePchFlags INTERFACE)
  target_link_libraries(
    SpectrePchFlags
    INTERFACE
    Blaze
    Brigand
    SpectreFlags
    )

  # Targets can reuse the PCH generated for this library. They must also link
  # the `SpectrePchFlags` library. Notes:
  # - We store the PCH header in ${CMAKE_SOURCE_DIR}/tools so that it is not
  #   accidentally included anywhere.
  # - We also store a source file that just includes the PCH header so the
  #   library has something to compile.
  # - This is an object library because we don't need to actually link it, we
  #   only need to build it so it generates the PCH.
  configure_file(
    ${CMAKE_SOURCE_DIR}/tools/SpectrePch.hpp
    ${CMAKE_BINARY_DIR}
    )
  set(SPECTRE_PCH_HEADER ${CMAKE_BINARY_DIR}/SpectrePch.hpp)
  configure_file(
    ${CMAKE_SOURCE_DIR}/tools/SpectrePch.cpp
    ${CMAKE_BINARY_DIR}
    )
  add_library(
    SpectrePch
    OBJECT
    ${CMAKE_BINARY_DIR}/SpectrePch.cpp)
  target_precompile_headers(
    SpectrePch
    PRIVATE
    ${CMAKE_BINARY_DIR}/SpectrePch.hpp
    )
  target_link_libraries(
    SpectrePch
    PRIVATE
    SpectrePchFlags
    )
endif (USE_PCH)
