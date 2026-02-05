# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Chebyshev.cpp
  Fourier.cpp
  Jacobi.cpp
  Legendre.cpp
  Zernike.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Chebyshev.hpp
  Fourier.hpp
  Jacobi.hpp
  Zernike.hpp
  )
