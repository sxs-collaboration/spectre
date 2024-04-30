# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  CartesianToSpherical.cpp
  FrameTransform.cpp
  OrthonormalOneform.cpp
  RaiseOrLowerIndex.cpp
  Trace.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  CartesianToSpherical.hpp
  CrossProduct.hpp
  Determinant.hpp
  DeterminantAndInverse.hpp
  DotProduct.hpp
  FrameTransform.hpp
  Magnitude.hpp
  Norms.hpp
  OrthonormalOneform.hpp
  OuterProduct.hpp
  RaiseOrLowerIndex.hpp
  Trace.hpp
  )

add_subdirectory(Python)
