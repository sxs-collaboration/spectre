// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Elasticity/Equations.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Elliptic/Systems/Elasticity/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
// IWYU pragma: no_forward_declare Tags::deriv
// IWYU pragma: no_forward_declare Tags::div
// IWYU pragma: no_forward_declare Variables
// IWYU pragma: no_forward_declare Tensor

namespace Elasticity {

template <size_t Dim>
void ComputeFirstOrderOperatorAction<Dim>::apply(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> operator_action,
    const gsl::not_null<tnsr::II<DataVector, Dim>*> auxiliary_operator_action,
    const tnsr::iJ<DataVector, Dim>& grad_displacement,
    const tnsr::II<DataVector, Dim>& stress, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
        inverse_jacobian,
    const tnsr::I<DataVector, Dim>& inertial_coords,
    const Elasticity::ConstitutiveRelations::ConstitutiveRelation<Dim>&
        constitutive_relation) noexcept {
  // Compute div(T)
  auto div_vars =
      make_with_value<Variables<tmpl::list<Tags::Stress<Dim>>>>(stress, 0.);
  get<Tags::Stress<Dim>>(div_vars) = stress;
  *operator_action = get<::Tags::div<Tags::Stress<Dim>>>(
      divergence(div_vars, mesh, inverse_jacobian));
  // Compute -\Y^{ijkl} S_{kl} - T^{ij}
  const auto computed_stress =
      constitutive_relation.stress(grad_displacement, inertial_coords);
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      auxiliary_operator_action->get(i, j) =
          computed_stress.get(i, j) - stress.get(i, j);
    }
  }
}

template <size_t Dim>
void ComputeFirstOrderNormalDotFluxes<Dim>::apply(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> normal_dot_flux,
    const gsl::not_null<tnsr::II<DataVector, Dim>*> auxiliary_normal_dot_flux,
    const tnsr::I<DataVector, Dim>& displacement,
    const tnsr::II<DataVector, Dim>& stress,
    const tnsr::i<DataVector, Dim>& interface_unit_normal,
    const tnsr::I<DataVector, Dim>& inertial_coords,
    const Elasticity::ConstitutiveRelations::ConstitutiveRelation<Dim>&
        constitutive_relation) noexcept {
  for (size_t j = 0; j < Dim; j++) {
    normal_dot_flux->get(j) = 0.;
    for (size_t i = 0; i < Dim; i++) {
      // The minus sign is to cancel the one in `lift_flux`
      normal_dot_flux->get(j) -=
          interface_unit_normal.get(i) * stress.get(i, j);
    }
  }

  auto normal_times_displacement =
      make_with_value<tnsr::iJ<DataVector, Dim>>(displacement, 0.);
  // The minus sign is to cancel the one in `lift_flux`
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      normal_times_displacement.get(i, j) =
          -1. * interface_unit_normal.get(i) * displacement.get(j);
    }
  }
  *auxiliary_normal_dot_flux =
      constitutive_relation.stress(normal_times_displacement, inertial_coords);
}

template <size_t Dim>
void FirstOrderInternalPenaltyFlux<Dim>::package_data(
    const gsl::not_null<Variables<package_tags>*> packaged_data,
    const tnsr::I<DataVector, Dim>& displacement,
    const tnsr::iJ<DataVector, Dim>& grad_displacement,
    const tnsr::i<DataVector, Dim>& interface_unit_normal,
    const tnsr::I<DataVector, Dim>& inertial_coords,
    const Elasticity::ConstitutiveRelations::ConstitutiveRelation<Dim>&
        constitutive_relation) const noexcept {
  auto normal_times_displacement =
      make_with_value<tnsr::iJ<DataVector, Dim>>(displacement, 0.);
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      normal_times_displacement.get(i, j) =
          interface_unit_normal.get(i) * displacement.get(j);
    }
  }
  get<AuxiliaryFlux>(*packaged_data) =
      constitutive_relation.stress(normal_times_displacement, inertial_coords);

  for (size_t j = 0; j < Dim; j++) {
    get<NormalDotAuxiliaryFlux>(*packaged_data).get(j) =
        get<0>(interface_unit_normal) *
        get<AuxiliaryFlux>(*packaged_data).get(0, j);
    for (size_t i = 1; i < Dim; i++) {
      get<NormalDotAuxiliaryFlux>(*packaged_data).get(j) +=
          interface_unit_normal.get(i) *
          get<AuxiliaryFlux>(*packaged_data).get(i, j);
    }
  }

  const auto computed_stress =
      constitutive_relation.stress(grad_displacement, inertial_coords);
  for (size_t j = 0; j < Dim; j++) {
    get<NormalDotStress>(*packaged_data).get(j) =
        get<0>(interface_unit_normal) * computed_stress.get(0, j);
    for (size_t i = 1; i < Dim; i++) {
      get<NormalDotStress>(*packaged_data).get(j) +=
          interface_unit_normal.get(i) * computed_stress.get(i, j);
    }
  }
}

template <size_t Dim>
void FirstOrderInternalPenaltyFlux<Dim>::operator()(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> numerical_flux,
    const gsl::not_null<tnsr::II<DataVector, Dim>*> auxiliary_numerical_flux,
    const tnsr::II<DataVector, Dim>& auxiliary_flux_interior,
    const tnsr::I<DataVector, Dim>& normal_dot_stress_interior,
    const tnsr::I<DataVector, Dim>& normal_dot_auxiliary_flux_interior,
    const tnsr::II<DataVector, Dim>& minus_auxiliary_flux_exterior,
    const tnsr::I<DataVector, Dim>& minus_normal_dot_stress_exterior,
    const tnsr::I<DataVector, Dim>& minus_normal_dot_auxiliary_flux_exterior)
    const noexcept {
  // Need polynomial degress and element size to compute this dynamically
  const double penalty = penalty_parameter_;

  // The outer minus sign is to cancel the one in `lift_flux`
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      auxiliary_numerical_flux->get(i, j) =
          -0.5 * (auxiliary_flux_interior.get(i, j) -
                  minus_auxiliary_flux_exterior.get(i, j));
    }
  }

  // The outer minus sign is to cancel the one in `lift_flux`
  for (size_t i = 0; i < Dim; i++) {
    numerical_flux->get(i) =
        -0.5 * (normal_dot_stress_interior.get(i) -
                minus_normal_dot_stress_exterior.get(i)) +
        penalty * (normal_dot_auxiliary_flux_interior.get(i) -
                   minus_normal_dot_auxiliary_flux_exterior.get(i));
  }
}

template <size_t Dim>
void FirstOrderInternalPenaltyFlux<Dim>::compute_dirichlet_boundary(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> numerical_flux,
    const gsl::not_null<tnsr::II<DataVector, Dim>*> auxiliary_numerical_flux,
    const tnsr::I<DataVector, Dim>& dirichlet_displacement,
    const tnsr::i<DataVector, Dim>& interface_unit_normal,
    const tnsr::I<DataVector, Dim>& inertial_coords,
    const Elasticity::ConstitutiveRelations::ConstitutiveRelation<Dim>&
        constitutive_relation) const noexcept {
  // Need polynomial degress and element size to compute this dynamically
  const double penalty = penalty_parameter_;

  auto normal_times_displacement =
      make_with_value<tnsr::iJ<DataVector, Dim>>(dirichlet_displacement, 0.);
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      normal_times_displacement.get(i, j) =
          interface_unit_normal.get(i) * dirichlet_displacement.get(j);
    }
  }
  const auto flux_for_stress =
      constitutive_relation.stress(normal_times_displacement, inertial_coords);

  // The minus sign is to cancel the one in `lift_flux`
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      auxiliary_numerical_flux->get(i, j) = -flux_for_stress.get(i, j);
    }
  }

  // The minus sign is to cancel the one in `lift_flux`
  for (size_t j = 0; j < Dim; j++) {
    numerical_flux->get(j) = 0.;
    for (size_t i = 0; i < Dim; i++) {
      numerical_flux->get(j) -= 2. * penalty * interface_unit_normal.get(i) *
                                flux_for_stress.get(i, j);
    }
  }
}

}  // namespace Elasticity

// Instantiate needed gradient and divergence templates
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"  // IWYU pragma: keep

template <size_t Dim>
using variables_tags =
    typename Elasticity::FirstOrderSystem<Dim>::variables_tag::type::tags_list;
template <size_t Dim>
using grad_tags = typename Elasticity::FirstOrderSystem<Dim>::gradient_tags;
template <size_t Dim>
using div_tags = typename Elasticity::FirstOrderSystem<Dim>::divergence_tags;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                             \
  template class Elasticity::ComputeFirstOrderOperatorAction<DIM(data)>;   \
  template class Elasticity::ComputeFirstOrderNormalDotFluxes<DIM(data)>;  \
  template class Elasticity::FirstOrderInternalPenaltyFlux<DIM(data)>;     \
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

GENERATE_INSTANTIATIONS(INSTANTIATION, (2, 3))

#undef INSTANTIATION
#undef DIM
