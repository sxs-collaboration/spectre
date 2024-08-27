// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Poisson/Equations.hpp"

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Poisson {

template <typename DataType, size_t Dim>
void flat_cartesian_fluxes(
    const gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
    const tnsr::i<DataType, Dim>& field_gradient) {
  for (size_t d = 0; d < Dim; d++) {
    flux_for_field->get(d) = field_gradient.get(d);
  }
}

template <typename DataType, size_t Dim>
void curved_fluxes(const gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
                   const tnsr::II<DataVector, Dim>& inv_spatial_metric,
                   const tnsr::i<DataType, Dim>& field_gradient) {
  raise_or_lower_index(flux_for_field, field_gradient, inv_spatial_metric);
}

template <typename DataType, size_t Dim>
void fluxes_on_face(gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
                    const tnsr::I<DataVector, Dim>& face_normal_vector,
                    const Scalar<DataType>& field) {
  std::copy(face_normal_vector.begin(), face_normal_vector.end(),
            flux_for_field->begin());
  for (size_t d = 0; d < Dim; d++) {
    flux_for_field->get(d) *= get(field);
  }
}

template <typename DataType, size_t Dim>
void add_curved_sources(const gsl::not_null<Scalar<DataType>*> source_for_field,
                        const tnsr::i<DataVector, Dim>& christoffel_contracted,
                        const tnsr::I<DataType, Dim>& flux_for_field) {
  get(*source_for_field) -=
      get(dot_product(christoffel_contracted, flux_for_field));
}

template <size_t Dim, typename DataType>
void Fluxes<Dim, Geometry::FlatCartesian, DataType>::apply(
    const gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
    const Scalar<DataType>& /*field*/,
    const tnsr::i<DataType, Dim>& field_gradient) {
  flat_cartesian_fluxes(flux_for_field, field_gradient);
}

template <size_t Dim, typename DataType>
void Fluxes<Dim, Geometry::FlatCartesian, DataType>::apply(
    const gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
    const tnsr::i<DataVector, Dim>& /*face_normal*/,
    const tnsr::I<DataVector, Dim>& face_normal_vector,
    const Scalar<DataType>& field) {
  fluxes_on_face(flux_for_field, face_normal_vector, field);
}

template <size_t Dim, typename DataType>
void Fluxes<Dim, Geometry::Curved, DataType>::apply(
    const gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
    const tnsr::II<DataVector, Dim>& inv_spatial_metric,
    const Scalar<DataType>& /*field*/,
    const tnsr::i<DataType, Dim>& field_gradient) {
  curved_fluxes(flux_for_field, inv_spatial_metric, field_gradient);
}

template <size_t Dim, typename DataType>
void Fluxes<Dim, Geometry::Curved, DataType>::apply(
    const gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
    const tnsr::II<DataVector, Dim>& /*inv_spatial_metric*/,
    const tnsr::i<DataVector, Dim>& /*face_normal*/,
    const tnsr::I<DataVector, Dim>& face_normal_vector,
    const Scalar<DataType>& field) {
  fluxes_on_face(flux_for_field, face_normal_vector, field);
}

template <size_t Dim, typename DataType>
void Sources<Dim, Geometry::Curved, DataType>::apply(
    const gsl::not_null<Scalar<DataType>*> equation_for_field,
    const tnsr::i<DataVector, Dim>& christoffel_contracted,
    const Scalar<DataType>& /*field*/,
    const tnsr::I<DataType, Dim>& field_flux) {
  add_curved_sources(equation_for_field, christoffel_contracted, field_flux);
}

}  // namespace Poisson

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                  \
  template void Poisson::flat_cartesian_fluxes(                               \
      const gsl::not_null<tnsr::I<DTYPE(data), DIM(data)>*>,                  \
      const tnsr::i<DTYPE(data), DIM(data)>&);                                \
  template void Poisson::curved_fluxes(                                       \
      const gsl::not_null<tnsr::I<DTYPE(data), DIM(data)>*>,                  \
      const tnsr::II<DataVector, DIM(data)>&,                                 \
      const tnsr::i<DTYPE(data), DIM(data)>&);                                \
  template void Poisson::fluxes_on_face(                                      \
      const gsl::not_null<tnsr::I<DTYPE(data), DIM(data)>*>,                  \
      const tnsr::I<DataVector, DIM(data)>&, const Scalar<DTYPE(data)>&);     \
  template void Poisson::add_curved_sources(                                  \
      const gsl::not_null<Scalar<DTYPE(data)>*>,                              \
      const tnsr::i<DataVector, DIM(data)>&,                                  \
      const tnsr::I<DTYPE(data), DIM(data)>&);                                \
  template class Poisson::Fluxes<DIM(data), Poisson::Geometry::FlatCartesian, \
                                 DTYPE(data)>;                                \
  template class Poisson::Fluxes<DIM(data), Poisson::Geometry::Curved,        \
                                 DTYPE(data)>;                                \
  template class Poisson::Sources<DIM(data), Poisson::Geometry::Curved,       \
                                  DTYPE(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector, ComplexDataVector), (1, 2, 3))

#undef INSTANTIATE
#undef DIM
