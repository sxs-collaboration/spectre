// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Poisson/Equations.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Poisson {

template <size_t Dim>
void euclidean_fluxes(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
    const tnsr::i<DataVector, Dim>& field_gradient) noexcept {
  for (size_t d = 0; d < Dim; d++) {
    flux_for_field->get(d) = field_gradient.get(d);
  }
}

template <size_t Dim>
void noneuclidean_fluxes(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
    const tnsr::II<DataVector, Dim>& inv_spatial_metric,
    const Scalar<DataVector>& det_spatial_metric,
    const tnsr::i<DataVector, Dim>& field_gradient) noexcept {
  raise_or_lower_index(flux_for_field, field_gradient, inv_spatial_metric);
  for (size_t i = 0; i < Dim; i++) {
    flux_for_field->get(i) *= sqrt(get(det_spatial_metric));
  }
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

}  // namespace Poisson

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                             \
  template void Poisson::euclidean_fluxes<DIM(data)>(                    \
      const gsl::not_null<tnsr::I<DataVector, DIM(data)>*>,              \
      const tnsr::i<DataVector, DIM(data)>&) noexcept;                   \
  template void Poisson::noneuclidean_fluxes<DIM(data)>(                 \
      const gsl::not_null<tnsr::I<DataVector, DIM(data)>*>,              \
      const tnsr::II<DataVector, DIM(data)>&, const Scalar<DataVector>&, \
      const tnsr::i<DataVector, DIM(data)>&) noexcept;                   \
  template void Poisson::auxiliary_fluxes<DIM(data)>(                    \
      gsl::not_null<tnsr::Ij<DataVector, DIM(data)>*>,                   \
      const Scalar<DataVector>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

// Instantiate derivative templates
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

template <size_t Dim>
using variables_tag = typename Poisson::FirstOrderSystem<Dim>::variables_tag;
template <size_t Dim>
using fluxes_tags_list = db::get_variables_tags_list<db::add_tag_prefix<
    ::Tags::Flux, variables_tag<Dim>, tmpl::size_t<Dim>, Frame::System>>;

#define INSTANTIATE_DERIVS(_, data)                                            \
  template Variables<db::wrap_tags_in<Tags::div, fluxes_tags_list<DIM(data)>>> \
  divergence<fluxes_tags_list<DIM(data)>, DIM(data), Frame::System>(           \
      const Variables<fluxes_tags_list<DIM(data)>>&, const Mesh<DIM(data)>&,   \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,      \
                            Frame::System>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_DERIVS, (1, 2, 3))

#undef INSTANTIATE
#undef INSTANTIATE_DERIVS
#undef DIM
