# Distributed under the MIT License.
# See LICENSE.txt for details.

option(ENABLE_PROFILING, "Enables various options to make profiling easier" OFF)

option(KEEP_FRAME_POINTER, "Add keep frame pointer for profiling" OFF)

add_library(Profiling::KeepFramePointer IMPORTED INTERFACE)
add_library(Profiling::EnableProfiling IMPORTED INTERFACE)

if (KEEP_FRAME_POINTER OR ENABLE_PROFILING)
  set_property(
    TARGET Profiling::KeepFramePointer
    APPEND PROPERTY
    INTERFACE_COMPILE_OPTIONS
    $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
    $<$<COMPILE_LANGUAGE:CXX>:-mno-omit-leaf-frame-pointer>
    )
endif()

if (ENABLE_PROFILING)
  set_property(
    TARGET Profiling::EnableProfiling
    APPEND PROPERTY
    INTERFACE_COMPILE_DEFINITIONS
    $<$<COMPILE_LANGUAGE:CXX>:SPECTRE_PROFILING>
    )
endif()

target_link_libraries(
  SpectreFlags
  INTERFACE
  Profiling::EnableProfiling
  Profiling::KeepFramePointer
  )
