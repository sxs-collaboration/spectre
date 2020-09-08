# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  IO
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ElementActions.hpp
  ElementDataReader.hpp
  ElementDataReaderActions.hpp
  ReadSpecThirdOrderPiecewisePolynomial.hpp
  Tags.hpp
  )
