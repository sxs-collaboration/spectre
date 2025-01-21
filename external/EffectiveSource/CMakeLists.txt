set(LIBRARY EffectiveSource)

add_library(${LIBRARY} kerr-circular.c)

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  GSL::gsl
  )

target_include_directories(${LIBRARY} SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
