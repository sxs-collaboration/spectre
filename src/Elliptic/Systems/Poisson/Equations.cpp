// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Poisson/Equations.hpp"

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Poisson {

template <size_t Dim>
void flat_cartesian_fluxes(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
    const tnsr::i<DataVector, Dim>& field_gradient) noexcept {
  for (size_t d = 0; d < Dim; d++) {
    flux_for_field->get(d) = field_gradient.get(d);
  }
}

template <size_t Dim>
void curved_fluxes(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
    const tnsr::II<DataVector, Dim>& inv_spatial_metric,
    const tnsr::i<DataVector, Dim>& field_gradient) noexcept {
  raise_or_lower_index(flux_for_field, field_gradient, inv_spatial_metric);
}

template <size_t Dim>
void add_curved_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_field,
    const tnsr::i<DataVector, Dim>& christoffel_contracted,
    const tnsr::I<DataVector, Dim>& flux_for_field) noexcept {
  get(*source_for_field) -=
      get(dot_product(christoffel_contracted, flux_for_field));
}

template <size_t Dim>
void auxiliary_fluxes(
    gsl::not_null<tnsr::Ij<DataVector, Dim>*> flux_for_gradient,
    const Scalar<DataVector>& field) noexcept {
  std::fill(flux_for_gradient->begin(), flux_for_gradient->end(), 0.);
  for (size_t d = 0; d < Dim; d++) {
    flux_for_gradient->get(d, d) = get(field);
  }
}

template <size_t Dim>
void Fluxes<Dim, Geometry::FlatCartesian>::apply(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
    const tnsr::i<DataVector, Dim>& field_gradient) noexcept {
  flat_cartesian_fluxes(flux_for_field, field_gradient);
}

template <size_t Dim>
void Fluxes<Dim, Geometry::Curved>::apply(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
    const tnsr::II<DataVector, Dim>& inv_spatial_metric,
    const tnsr::i<DataVector, Dim>& field_gradient) noexcept {
  curved_fluxes(flux_for_field, inv_spatial_metric, field_gradient);
}

template <size_t Dim>
void Fluxes<Dim, Geometry::FlatCartesian>::apply(
    const gsl::not_null<tnsr::Ij<DataVector, Dim>*> flux_for_gradient,
    const Scalar<DataVector>& field) noexcept {
  auxiliary_fluxes(flux_for_gradient, field);
}

template <size_t Dim>
void Fluxes<Dim, Geometry::Curved>::apply(
    const gsl::not_null<tnsr::Ij<DataVector, Dim>*> flux_for_gradient,
    const tnsr::II<DataVector, Dim>& /*inv_spatial_metric*/,
    const Scalar<DataVector>& field) noexcept {
  auxiliary_fluxes(flux_for_gradient, field);
}

template <size_t Dim>
void Sources<Dim, Geometry::FlatCartesian>::apply(
    const gsl::not_null<Scalar<DataVector>*> /*equation_for_field*/,
    const Scalar<DataVector>& /*field*/,
    const tnsr::I<DataVector, Dim>& /*field_flux*/) noexcept {}

template <size_t Dim>
void Sources<Dim, Geometry::Curved>::apply(
    const gsl::not_null<Scalar<DataVector>*> equation_for_field,
    const tnsr::i<DataVector, Dim>& christoffel_contracted,
    const Scalar<DataVector>& /*field*/,
    const tnsr::I<DataVector, Dim>& field_flux) noexcept {
  add_curved_sources(equation_for_field, christoffel_contracted, field_flux);
}

template <size_t Dim>
void Sources<Dim, Geometry::FlatCartesian>::apply(
    const gsl::not_null<
        tnsr::i<DataVector, Dim>*> /*equation_for_field_gradient*/,
    const Scalar<DataVector>& /*field*/) noexcept {}

template <size_t Dim>
void Sources<Dim, Geometry::Curved>::apply(
    const gsl::not_null<
        tnsr::i<DataVector, Dim>*> /*equation_for_field_gradient*/,
    const tnsr::i<DataVector, Dim>& /*christoffel_contracted*/,
    const Scalar<DataVector>& /*field*/) noexcept {}

}  // namespace Poisson

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void Poisson::flat_cartesian_fluxes<DIM(data)>(                     \
      const gsl::not_null<tnsr::I<DataVector, DIM(data)>*>,                    \
      const tnsr::i<DataVector, DIM(data)>&) noexcept;                         \
  template void Poisson::curved_fluxes<DIM(data)>(                             \
      const gsl::not_null<tnsr::I<DataVector, DIM(data)>*>,                    \
      const tnsr::II<DataVector, DIM(data)>&,                                  \
      const tnsr::i<DataVector, DIM(data)>&) noexcept;                         \
  template void Poisson::add_curved_sources<DIM(data)>(                        \
      const gsl::not_null<Scalar<DataVector>*>,                                \
      const tnsr::i<DataVector, DIM(data)>&,                                   \
      const tnsr::I<DataVector, DIM(data)>&) noexcept;                         \
  template void Poisson::auxiliary_fluxes<DIM(data)>(                          \
      gsl::not_null<tnsr::Ij<DataVector, DIM(data)>*>,                         \
      const Scalar<DataVector>&) noexcept;                                     \
  template class Poisson::Fluxes<DIM(data), Poisson::Geometry::FlatCartesian>; \
  template class Poisson::Fluxes<DIM(data), Poisson::Geometry::Curved>;        \
  template class Poisson::Sources<DIM(data),                                   \
                                  Poisson::Geometry::FlatCartesian>;           \
  template class Poisson::Sources<DIM(data), Poisson::Geometry::Curved>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
