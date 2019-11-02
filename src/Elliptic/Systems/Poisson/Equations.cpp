// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Poisson/Equations.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
// IWYU pragma: no_forward_declare Tags::deriv
// IWYU pragma: no_forward_declare Tags::div
// IWYU pragma: no_forward_declare Variables
// IWYU pragma: no_forward_declare Tensor

namespace Poisson {

template <size_t Dim>
void euclidean_fluxes(
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Physical>*>
        flux_for_field,
    const tnsr::i<DataVector, Dim, Frame::Physical>& field_gradient) noexcept {
  for (size_t d = 0; d < Dim; d++) {
    flux_for_field->get(d) = field_gradient.get(d);
  }
}

template <size_t Dim>
void noneuclidean_fluxes(
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Physical>*>
        flux_for_field,
    const tnsr::II<DataVector, Dim, Frame::Physical>& inv_spatial_metric,
    const Scalar<DataVector>& det_spatial_metric,
    const tnsr::i<DataVector, Dim, Frame::Physical>& field_gradient) noexcept {
  raise_or_lower_index(flux_for_field, field_gradient, inv_spatial_metric);
  for (size_t i = 0; i < Dim; i++) {
    flux_for_field->get(i) *= sqrt(get(det_spatial_metric));
  }
}

template <size_t Dim>
void auxiliary_fluxes(gsl::not_null<tnsr::Ij<DataVector, Dim, Frame::Physical>*>
                          flux_for_gradient,
                      const Scalar<DataVector>& field) noexcept {
  std::fill(flux_for_gradient->begin(), flux_for_gradient->end(), 0.);
  for (size_t d = 0; d < Dim; d++) {
    flux_for_gradient->get(d, d) = get(field);
  }
}

template <size_t Dim>
void FirstOrderInternalPenaltyFlux<Dim>::package_data(
    const gsl::not_null<Variables<package_tags>*> packaged_data,
    const Scalar<DataVector>& field,
    const tnsr::i<DataVector, Dim, Frame::Physical>& grad_field,
    const tnsr::i<DataVector, Dim, Frame::Physical>& interface_unit_normal)
    const noexcept {
  get<LinearSolver::Tags::Operand<Tags::Field>>(*packaged_data) = field;

  for (size_t d = 0; d < Dim; d++) {
    get<NormalTimesFieldFlux>(*packaged_data).get(d) =
        interface_unit_normal.get(d) * get(field);
  }

  get<NormalDotGradFieldFlux>(*packaged_data).get() =
      get<0>(interface_unit_normal) * get<0>(grad_field);
  for (size_t d = 1; d < Dim; d++) {
    get<NormalDotGradFieldFlux>(*packaged_data).get() +=
        interface_unit_normal.get(d) * grad_field.get(d);
  }
}

template <size_t Dim>
void FirstOrderInternalPenaltyFlux<Dim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> numerical_flux_for_field,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Physical>*>
        numerical_flux_for_auxiliary_field,
    const Scalar<DataVector>& field_interior,
    const tnsr::i<DataVector, Dim, Frame::Physical>&
        normal_times_field_interior,
    const Scalar<DataVector>& normal_dot_grad_field_interior,
    const Scalar<DataVector>& field_exterior,
    const tnsr::i<DataVector, Dim, Frame::Physical>&
        minus_normal_times_field_exterior,
    const Scalar<DataVector>& minus_normal_dot_grad_field_exterior) const
    noexcept {
  // Need polynomial degress and element size to compute this dynamically
  const double penalty = penalty_parameter_;

  // The minus sign in the equation is canceled by the one in `lift_flux`
  for (size_t d = 0; d < Dim; d++) {
    numerical_flux_for_auxiliary_field->get(d) =
        0.5 * (normal_times_field_interior.get(d) -
               minus_normal_times_field_exterior.get(d));
  }

  // The minus sign in the equation is canceled by the one in `lift_flux`
  numerical_flux_for_field->get() =
      0.5 * (get(normal_dot_grad_field_interior) -
             get(minus_normal_dot_grad_field_exterior)) -
      penalty * (get(field_interior) - get(field_exterior));
}

template <size_t Dim>
void FirstOrderInternalPenaltyFlux<Dim>::compute_dirichlet_boundary(
    const gsl::not_null<Scalar<DataVector>*> numerical_flux_for_field,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Physical>*>
        numerical_flux_for_auxiliary_field,
    const Scalar<DataVector>& dirichlet_field,
    const tnsr::i<DataVector, Dim, Frame::Physical>& interface_unit_normal)
    const noexcept {
  // Need polynomial degress and element size to compute this dynamically
  const double penalty = penalty_parameter_;

  // The minus sign in the equation is canceled by the one in `lift_flux`
  for (size_t d = 0; d < Dim; d++) {
    numerical_flux_for_auxiliary_field->get(d) =
        interface_unit_normal.get(d) * get(dirichlet_field);
  }

  // The minus sign in the equation is canceled by the one in `lift_flux`
  numerical_flux_for_field->get() = 2. * penalty * get(dirichlet_field);
}

}  // namespace Poisson

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template class Poisson::FirstOrderInternalPenaltyFlux<DIM(data)>;          \
  template void Poisson::euclidean_fluxes<DIM(data)>(                        \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Physical>*>, \
      const tnsr::i<DataVector, DIM(data), Frame::Physical>&) noexcept;      \
  template void Poisson::noneuclidean_fluxes<DIM(data)>(                     \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Physical>*>, \
      const tnsr::II<DataVector, DIM(data), Frame::Physical>&,               \
      const Scalar<DataVector>&,                                             \
      const tnsr::i<DataVector, DIM(data), Frame::Physical>&) noexcept;      \
  template void Poisson::auxiliary_fluxes<DIM(data)>(                        \
      gsl::not_null<tnsr::Ij<DataVector, DIM(data), Frame::Physical>*>,      \
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
    ::Tags::Flux, variables_tag<Dim>, tmpl::size_t<Dim>, Frame::Physical>>;

#define INSTANTIATE_DERIVS(_, data)                                            \
  template Variables<db::wrap_tags_in<Tags::div, fluxes_tags_list<DIM(data)>>> \
  divergence<fluxes_tags_list<DIM(data)>, DIM(data), Frame::Physical>(         \
      const Variables<fluxes_tags_list<DIM(data)>>&, const Mesh<DIM(data)>&,   \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,             \
                            Frame::Physical>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_DERIVS, (1, 2, 3))

#undef INSTANTIATE
#undef INSTANTIATE_DERIVS
#undef DIM
