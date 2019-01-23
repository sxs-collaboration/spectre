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
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
// IWYU pragma: no_forward_declare Tags::deriv
// IWYU pragma: no_forward_declare Tags::div
// IWYU pragma: no_forward_declare Variables
// IWYU pragma: no_forward_declare Tensor

namespace Poisson {

template <size_t Dim>
void ComputeFirstOrderOperatorAction<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> operator_for_field_source,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        operator_for_auxiliary_field_source,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& grad_field,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& auxiliary_field,
    const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
        inverse_jacobian) noexcept {
  auto div_vars = make_with_value<Variables<tmpl::list<AuxiliaryField<Dim>>>>(
      auxiliary_field, 0.);
  get<AuxiliaryField<Dim>>(div_vars) = auxiliary_field;
  // Tensors don't support math operations yet, so we have to `get` the
  // DataVector for the sign flip
  get(*operator_for_field_source) =
      -1. * get(get<Tags::div<AuxiliaryField<Dim>>>(
                divergence(div_vars, mesh, inverse_jacobian)));
  for (size_t d = 0; d < Dim; d++) {
    operator_for_auxiliary_field_source->get(d) =
        grad_field.get(d) - auxiliary_field.get(d);
  }
}

template <size_t Dim>
void ComputeFirstOrderNormalDotFluxes<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> normal_dot_flux_for_field,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        normal_dot_flux_for_auxiliary_field,
    const Scalar<DataVector>& field,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& auxiliary_field,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        interface_unit_normal) noexcept {
  // The minus sign in the equation is canceled by the one in `lift_flux`
  get(*normal_dot_flux_for_field) =
      get<0>(interface_unit_normal) * get<0>(auxiliary_field);
  for (size_t d = 1; d < Dim; d++) {
    get(*normal_dot_flux_for_field) +=
        interface_unit_normal.get(d) * auxiliary_field.get(d);
  }

  // The minus sign is to cancel the one in `lift_flux`
  for (size_t d = 0; d < Dim; d++) {
    normal_dot_flux_for_auxiliary_field->get(d) =
        -interface_unit_normal.get(d) * get(field);
  }
}

template <size_t Dim>
void FirstOrderInternalPenaltyFlux<Dim>::package_data(
    const gsl::not_null<Variables<package_tags>*> packaged_data,
    const Scalar<DataVector>& field,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& grad_field,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
    const noexcept {
  get<LinearSolver::Tags::Operand<Field>>(*packaged_data) = field;

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
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        numerical_flux_for_auxiliary_field,
    const Scalar<DataVector>& field_interior,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        normal_times_field_interior,
    const Scalar<DataVector>& normal_dot_grad_field_interior,
    const Scalar<DataVector>& field_exterior,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        minus_normal_times_field_exterior,
    const Scalar<DataVector>& minus_normal_dot_grad_field_exterior) const
    noexcept {
  // Need polynomial degress and element size to compute this dynamically
  const double penalty = penalty_parameter_;

  // The minus sign is to cancel the one in `lift_flux`
  for (size_t d = 0; d < Dim; d++) {
    numerical_flux_for_auxiliary_field->get(d) =
        -0.5 * (normal_times_field_interior.get(d) -
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
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        numerical_flux_for_auxiliary_field,
    const Scalar<DataVector>& dirichlet_field,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
    const noexcept {
  // Need polynomial degress and element size to compute this dynamically
  const double penalty = penalty_parameter_;

  // The minus sign is to cancel the one in `lift_flux`
  for (size_t d = 0; d < Dim; d++) {
    numerical_flux_for_auxiliary_field->get(d) =
        -interface_unit_normal.get(d) * get(dirichlet_field);
  }

  // The minus sign in the equation is canceled by the one in `lift_flux`
  numerical_flux_for_field->get() = 2. * penalty * get(dirichlet_field);
}

}  // namespace Poisson

// Instantiate needed gradient and divergence templates
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"  // IWYU pragma: keep

template <size_t Dim>
using variables_tags =
    typename Poisson::FirstOrderSystem<Dim>::variables_tag::type::tags_list;
template <size_t Dim>
using grad_tags = typename Poisson::FirstOrderSystem<Dim>::gradient_tags;
template <size_t Dim>
using div_tags = typename Poisson::FirstOrderSystem<Dim>::divergence_tags;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                             \
  template class Poisson::ComputeFirstOrderOperatorAction<DIM(data)>;      \
  template class Poisson::ComputeFirstOrderNormalDotFluxes<DIM(data)>;     \
  template class Poisson::FirstOrderInternalPenaltyFlux<DIM(data)>;        \
  template Variables<                                                      \
      db::wrap_tags_in<Tags::deriv, grad_tags<DIM(data)>,                  \
                       tmpl::size_t<DIM(data)>, Frame::Inertial>>          \
  partial_derivatives<grad_tags<DIM(data)>, variables_tags<DIM(data)>,     \
                      DIM(data), Frame::Inertial>(                         \
      const Variables<variables_tags<DIM(data)>>&, const Mesh<DIM(data)>&, \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,         \
                            Frame::Inertial>&) noexcept;                   \
  template Variables<db::wrap_tags_in<Tags::div, div_tags<DIM(data)>>>     \
  divergence<div_tags<DIM(data)>, DIM(data), Frame::Inertial>(             \
      const Variables<div_tags<DIM(data)>>&, const Mesh<DIM(data)>&,       \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,         \
                            Frame::Inertial>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
