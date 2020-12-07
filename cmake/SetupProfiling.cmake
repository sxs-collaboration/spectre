# Distributed under the MIT License.
# See LICENSE.txt for details.

option(KEEP_FRAME_POINTER, "Add keep frame pointer for profiling" OFF)

add_library(Profiling::KeepFramePointer IMPORTED INTERFACE)

if (KEEP_FRAME_POINTER)
  set_property(
    TARGET Profiling::KeepFramePointer
    APPEND PROPERTY
    INTERFACE_COMPILE_OPTIONS
    $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
    $<$<COMPILE_LANGUAGE:CXX>:-mno-omit-leaf-frame-pointer>
    )
endif()

target_link_libraries(
  SpectreFlags
  INTERFACE
  Profiling::KeepFramePointer
  )
