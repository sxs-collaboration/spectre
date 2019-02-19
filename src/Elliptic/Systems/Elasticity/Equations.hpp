// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"  // IWYU pragma: keep
#include "Options/Options.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tags::deriv
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

/// \cond
class DataVector;
template <size_t>
class Mesh;
namespace Tags {
template <typename>
struct Normalized;
}  // namespace Tags
namespace LinearSolver {
namespace Tags {
template <typename>
struct Operand;
}  // namespace Tags
}  // namespace LinearSolver
namespace Elasticity {
namespace Tags {
template <size_t Dim>
struct Displacement;
template <size_t Dim>
struct Stress;
}  // namespace Tags
}  // namespace Elasticity
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Elasticity {

/*!
 * \brief The bulk contribution to the linear operator action for the
 * first-order formulation of the Elasticity equations.
 *
 * \details The bulk contribution for the equation sourced by
 * \f$f_\mathrm{ext}^j\f$ is \f$\nabla_i T^{ij}\f$ and the one for the auxiliary
 * equation isn \f$-Y^{ijkl}\nabla_{(k}u_{l)} - T^{ij}\f$ (see
 * `Elasticity::FirstOrderSystem`)
 */
template <size_t Dim>
struct ComputeFirstOrderOperatorAction {
  using argument_tags = tmpl::list<
      ::Tags::deriv<LinearSolver::Tags::Operand<Tags::Displacement<Dim>>,
                    tmpl::size_t<Dim>, Frame::Inertial>,
      LinearSolver::Tags::Operand<Tags::Stress<Dim>>, ::Tags::Mesh<Dim>,
      ::Tags::InverseJacobian<::Tags::ElementMap<Dim>,
                              ::Tags::Coordinates<Dim, Frame::Logical>>,
      ::Tags::Coordinates<Dim, Frame::Inertial>>;
  using const_global_cache_tags =
      tmpl::list<Elasticity::Tags::ConstitutiveRelationBase>;
  static void apply(
      gsl::not_null<tnsr::I<DataVector, Dim>*> operator_action,
      gsl::not_null<tnsr::II<DataVector, Dim>*> auxiliary_operator_action,
      const tnsr::iJ<DataVector, Dim>& grad_displacement,
      const tnsr::II<DataVector, Dim>& stress, const Mesh<Dim>& mesh,
      const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
          inverse_jacobian,
      const tnsr::I<DataVector, Dim>& inertial_coords,
      const Elasticity::ConstitutiveRelations::ConstitutiveRelation<Dim>&
          constitutive_relation) noexcept;
};

/*
 * \brief The interface normal dotted into the fluxes for the first order
 * formulation of the Elasticity equations.
 *
 * \details For the sourced equation this is \f$n_i T^{ij}\f$ and for the
 * auxiliary equation it is \f$-Y^{ijkl}\n_{(k}u_{l)}\f$ (see
 * `Elsaticity::FirstOrderSystem` and `dg::lift_flux`).
 */
template <size_t Dim>
struct ComputeFirstOrderNormalDotFluxes {
  using argument_tags =
      tmpl::list<LinearSolver::Tags::Operand<Tags::Displacement<Dim>>,
                 LinearSolver::Tags::Operand<Tags::Stress<Dim>>,
                 ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim>>,
                 ::Tags::Coordinates<Dim, Frame::Inertial>>;
  using const_global_cache_tags =
      tmpl::list<Elasticity::Tags::ConstitutiveRelationBase>;
  static void apply(
      gsl::not_null<tnsr::I<DataVector, Dim>*> normal_dot_flux,
      gsl::not_null<tnsr::II<DataVector, Dim>*> auxiliary_normal_dot_flux,
      const tnsr::I<DataVector, Dim>& displacement,
      const tnsr::II<DataVector, Dim>& stress,
      const tnsr::i<DataVector, Dim>& interface_unit_normal,
      const tnsr::I<DataVector, Dim>& inertial_coords,
      const Elasticity::ConstitutiveRelations::ConstitutiveRelation<Dim>&
          constitutive_relation) noexcept;
};

/*!
 * \brief The internal penalty flux for the first oder formulation of the
 * Elasticity equations.
 *
 * \details For the sourced equation this is
 * \f[
 * {T^*}^{ij}=\frac{1}{2}\left( -Y^{ijkl}\nabla_{(k} u_{l)}^\mathrm{int} -
 * Y^{ijkl}\nabla_{(k)}u_{l)}^\mathrm{ext} \right) +
 * \sigma \left(-Y^{ijkl}n_{(k}u_{l)}^\mathrm{int} +
 * Y^{ijkl}n_{(k}u_{l)}^\mathrm{ext}\right)
 * \f]
 * and for the auxiliary equation it is
 * \f[
 * {u^*}_i=\frac{ u_i^\mathrm{int} + u_i^\mathrm{ext}}{2} \text{.}
 * \f]
 * The penalty factor \f$\sigma\f$ is responsible for removing zero eigenmodes
 * and impacts the conditioning of the linear operator to be solved. It can be
 * chosen as \f$\sigma=C\frac{N_\mathrm{points}^2}{h}\f$ where
 * \f$N_\mathrm{points}\f$ is the number of collocation points (i.e. the
 * polynomial degree plus 1), \f$h\f$ is a measure of the element size in
 * inertial coordinates and \f$C\geq 1\f$ is a free parameter (see e.g.
 * \cite HesthavenWarburton, section 7.2).
 */
template <size_t Dim>
struct FirstOrderInternalPenaltyFlux {
 public:
  struct PenaltyParameter {
    using type = double;
    static constexpr OptionString help = {
        "The prefactor to the penalty term of the flux."};
  };
  using options = tmpl::list<PenaltyParameter>;
  static constexpr OptionString help = {
      "Computes the internal penalty flux for an elasticity system."};

  FirstOrderInternalPenaltyFlux() = default;
  explicit FirstOrderInternalPenaltyFlux(double penalty_parameter)
      : penalty_parameter_(penalty_parameter) {}

  // clang-tidy: non-const reference
  void pup(PUP::er& p) noexcept { p | penalty_parameter_; }  // NOLINT

  // -Y^{ijkl} n_{(k} {u^*}^{l)}
  struct AuxiliaryFlux : db::SimpleTag {
    using type = tnsr::II<DataVector, Dim>;
    static std::string name() noexcept { return "AuxiliaryFlux"; }
  };

  // -n_i Y^{ijkl} n_{(k} {u^*}^{l)}
  struct NormalDotAuxiliaryFlux : db::SimpleTag {
    using type = tnsr::I<DataVector, Dim>;
    static std::string name() noexcept { return "NormalDotAuxiliaryFlux"; }
  };

  // -n_i Y^{ijkl} \nabla_{(k} u_{l)}
  struct NormalDotStress : db::SimpleTag {
    using type = tnsr::I<DataVector, Dim>;
    static std::string name() noexcept { return "NormalDotStress"; }
  };

  using argument_tags = tmpl::list<
      LinearSolver::Tags::Operand<Tags::Displacement<Dim>>,
      ::Tags::deriv<LinearSolver::Tags::Operand<Tags::Displacement<Dim>>,
                    tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim>>,
      ::Tags::Coordinates<Dim, Frame::Inertial>>;
  using const_global_cache_tags =
      tmpl::list<Elasticity::Tags::ConstitutiveRelationBase>;

  using package_tags =
      tmpl::list<AuxiliaryFlux, NormalDotStress, NormalDotAuxiliaryFlux>;

  void package_data(
      gsl::not_null<Variables<package_tags>*> packaged_data,
      const tnsr::I<DataVector, Dim>& displacement,
      const tnsr::iJ<DataVector, Dim>& grad_displacement,
      const tnsr::i<DataVector, Dim>& interface_unit_normal,
      const tnsr::I<DataVector, Dim>& inertial_coords,
      const Elasticity::ConstitutiveRelations::ConstitutiveRelation<Dim>&
          constitutive_relation) const noexcept;

  void operator()(
      gsl::not_null<tnsr::I<DataVector, Dim>*> numerical_flux,
      gsl::not_null<tnsr::II<DataVector, Dim>*> auxiliary_numerical_flux,
      const tnsr::II<DataVector, Dim>& auxiliary_flux_interior,
      const tnsr::I<DataVector, Dim>& normal_dot_stress_interior,
      const tnsr::I<DataVector, Dim>& normal_dot_auxiliary_flux_interior,
      const tnsr::II<DataVector, Dim>& minus_auxiliary_flux_exterior,
      const tnsr::I<DataVector, Dim>& minus_normal_dot_stress_exterior,
      const tnsr::I<DataVector, Dim>& minus_normal_dot_auxiliary_flux_exterior)
      const noexcept;

  using boundary_argument_tags =
      tmpl::list<::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim>>,
                 ::Tags::Coordinates<Dim, Frame::Inertial>>;

  void compute_dirichlet_boundary(
      gsl::not_null<tnsr::I<DataVector, Dim>*> numerical_flux,
      gsl::not_null<tnsr::II<DataVector, Dim>*> auxiliary_numerical_flux,
      const tnsr::I<DataVector, Dim>& dirichlet_displacement,
      const tnsr::i<DataVector, Dim>& interface_unit_normal,
      const tnsr::I<DataVector, Dim>& inertial_coords,
      const Elasticity::ConstitutiveRelations::ConstitutiveRelation<Dim>&
          constitutive_relation) const noexcept;

 private:
  double penalty_parameter_{};
};

}  // namespace Elasticity
