// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/GeodesicAcceleration.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace gr {
template <typename DataType, size_t Dim>
void geodesic_acceleration(
    gsl::not_null<tnsr::I<DataType, Dim>*> acceleration,
    const tnsr::I<DataType, Dim>& velocity,
    const tnsr::Abb<DataType, Dim>& christoffel_second_kind) {
  for (size_t i = 0; i < Dim; ++i) {
    acceleration->get(i) =
        velocity.get(i) * christoffel_second_kind.get(0, 0, 0) -
        christoffel_second_kind.get(i + 1, 0, 0);
    for (size_t j = 0; j < Dim; ++j) {
      acceleration->get(i) +=
          2. * velocity.get(j) *
          (velocity.get(i) * christoffel_second_kind.get(0, j + 1, 0) -
           christoffel_second_kind.get(i + 1, j + 1, 0));
      for (size_t k = 0; k < Dim; ++k) {
        acceleration->get(i) +=
            velocity.get(j) * velocity.get(k) *
            (velocity.get(i) * christoffel_second_kind.get(0, j + 1, k + 1) -
             christoffel_second_kind.get(i + 1, j + 1, k + 1));
      }
    }
  }
}

template <typename DataType, size_t Dim>
tnsr::I<DataType, Dim> geodesic_acceleration(
    const tnsr::I<DataType, Dim>& velocity,
    const tnsr::Abb<DataType, Dim>& christoffel_second_kind) {
  tnsr::I<DataType, Dim> acceleration{get_size(get<0>(velocity))};
  geodesic_acceleration(make_not_null(&acceleration), velocity,
                        christoffel_second_kind);
  return acceleration;
}

}  // namespace gr
template void gr::geodesic_acceleration(
    const gsl::not_null<tnsr::I<double, 3>*> acceleration,
    const tnsr::I<double, 3>& velocity,
    const tnsr::Abb<double, 3>& christoffel_second_kind);
template tnsr::I<double, 3> gr::geodesic_acceleration(
    const tnsr::I<double, 3>& velocity,
    const tnsr::Abb<double, 3>& christoffel_second_kind);
template void gr::geodesic_acceleration(
    const gsl::not_null<tnsr::I<DataVector, 3>*> acceleration,
    const tnsr::I<DataVector, 3>& velocity,
    const tnsr::Abb<DataVector, 3>& christoffel_second_kind);
template tnsr::I<DataVector, 3> gr::geodesic_acceleration(
    const tnsr::I<DataVector, 3>& velocity,
    const tnsr::Abb<DataVector, 3>& christoffel_second_kind);
