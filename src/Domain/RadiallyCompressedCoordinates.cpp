// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/RadiallyCompressedCoordinates.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain {

template <typename DataType, size_t Dim, typename CoordsFrame>
void radially_compressed_coordinates(
    const gsl::not_null<tnsr::I<DataType, Dim, CoordsFrame>*> result,
    const tnsr::I<DataType, Dim, CoordsFrame>& coordinates,
    const double inner_radius, const double outer_radius,
    const CoordinateMaps::Distribution compression) {
  const DataType radius = get(magnitude(coordinates));
  // Return early if all radii are within the inner radius
  if (max(radius) <= inner_radius + 1.e-14) {
    *result = coordinates;
    return;
  }
  // The compressed outer radius is chosen so it increases with both the inner
  // and the outer radius but exponentials are tamed
  const double compressed_outer_radius = inner_radius * log10(outer_radius);
  // We use the inverse of the Interval map, which is also used to distribute
  // grid points radially
  CoordinateMaps::Interval interval{inner_radius, compressed_outer_radius,
                                    inner_radius, outer_radius,
                                    compression,  0.};
  DataType compressed_radius = radius;
  for (size_t i = 0; i < get_size(radius); ++i) {
    if (get_element(radius, i) > inner_radius + 1.e-14) {
      get_element(compressed_radius, i) =
          interval.inverse({{get_element(radius, i)}}).value()[0];
    }
  }
  *result = coordinates;
  for (size_t d = 0; d < Dim; ++d) {
    result->get(d) *= compressed_radius / radius;
  }
}

void RadiallyCompressedCoordinatesOptions::pup(PUP::er& p) {
  p | inner_radius;
  p | outer_radius;
  p | compression;
}

template <typename DataType, size_t Dim, typename CoordsFrame>
tnsr::I<DataType, Dim, CoordsFrame> radially_compressed_coordinates(
    const tnsr::I<DataType, Dim, CoordsFrame>& coordinates,
    const double inner_radius, const double outer_radius,
    const CoordinateMaps::Distribution compression) {
  tnsr::I<DataType, Dim, CoordsFrame> result{};
  radially_compressed_coordinates(make_not_null(&result), coordinates,
                                  inner_radius, outer_radius, compression);
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                               \
  template void radially_compressed_coordinates(                           \
      gsl::not_null<tnsr::I<DTYPE(data), DIM(data), FRAME(data)>*> result, \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& coordinates,     \
      double inner_radius, double outer_radius,                            \
      CoordinateMaps::Distribution compression);                           \
  template tnsr::I<DTYPE(data), DIM(data), FRAME(data)>                    \
  radially_compressed_coordinates(                                         \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& coordinates,     \
      double inner_radius, double outer_radius,                            \
      CoordinateMaps::Distribution compression);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector), (1, 2, 3),
                        (Frame::Inertial))

#undef DTYPE
#undef DIM
#undef FRAME
#undef INSTANTIATE

}  // namespace domain
