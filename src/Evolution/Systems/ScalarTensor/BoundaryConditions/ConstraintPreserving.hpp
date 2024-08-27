// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/ConstraintPreservingSphericalRadiation.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Bjorhus.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/ScalarTensor/Tags.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarTensor::BoundaryConditions {
/*!
 * \brief Sets constraint-preserving boundary conditions on the variables of the
 * ScalarTensor system.
 * \details The constraint-preserving boundary conditions on the scalar
 * variables are approximate as they assume a fixed spacetime geometry.
 * Likewise, the constraint-preserving boundary conditions on the metric
 * variables assume that there is no back-reaction of the scalar stress energy
 * tensor on the metric.
 *
 */
class ConstraintPreserving final : public BoundaryCondition {
 public:
  using options = tmpl::push_back<
      typename gh::BoundaryConditions::ConstraintPreservingBjorhus<3>::options>;

  static constexpr Options::String help{
      "Constraint-preserving boundary conditions are applied for the "
      "Generalized Harmonic variables and spherical radiation constraint-"
      "preserving boundary conditions are applied for the scalar variables."};

  ConstraintPreserving() = default;
  explicit ConstraintPreserving(
      gh::BoundaryConditions::detail::ConstraintPreservingBjorhusType type);

  ConstraintPreserving(ConstraintPreserving&&) = default;
  ConstraintPreserving& operator=(ConstraintPreserving&&) = default;
  ConstraintPreserving(const ConstraintPreserving&) = default;
  ConstraintPreserving& operator=(const ConstraintPreserving&) = default;
  ~ConstraintPreserving() override = default;

  explicit ConstraintPreserving(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, ConstraintPreserving);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::GhostAndTimeDerivative;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags =
      tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
                 gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>,
                 CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi,
                 CurvedScalarWave::Tags::Phi<3>>;
  using dg_interior_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>,
                 ::gh::ConstraintDamping::Tags::ConstraintGamma1,
                 ::gh::ConstraintDamping::Tags::ConstraintGamma2,
                 gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 gr::Tags::InverseSpatialMetric<DataVector, 3>,
                 gr::Tags::InverseSpacetimeMetric<DataVector, 3>,
                 gr::Tags::SpacetimeNormalVector<DataVector, 3>,
                 gh::Tags::ThreeIndexConstraint<DataVector, 3>,
                 gh::Tags::GaugeH<DataVector, 3>,
                 gh::Tags::SpacetimeDerivGaugeH<DataVector, 3>,
                 CurvedScalarWave::Tags::ConstraintGamma1,
                 CurvedScalarWave::Tags::ConstraintGamma2>;
  using dg_interior_primitive_variables_tags = tmpl::list<>;
  using dg_gridless_tags = tmpl::list<>;

  static std::optional<std::string> dg_ghost(
      gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> spacetime_metric,
      gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> pi,
      gsl::not_null<tnsr::iaa<DataVector, 3, Frame::Inertial>*> phi,

      gsl::not_null<Scalar<DataVector>*> psi_scalar,
      gsl::not_null<Scalar<DataVector>*> pi_scalar,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> phi_scalar,

      // c.f. dg_package_data_temporary_tags from the combined Upwind correction
      gsl::not_null<Scalar<DataVector>*> gamma1,
      gsl::not_null<Scalar<DataVector>*> gamma2,
      gsl::not_null<Scalar<DataVector>*> lapse,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
      gsl::not_null<Scalar<DataVector>*> gamma1_scalar,
      gsl::not_null<Scalar<DataVector>*> gamma2_scalar,

      gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector,

      const tnsr::aa<DataVector, 3, Frame::Inertial>& interior_spacetime_metric,
      const tnsr::aa<DataVector, 3, Frame::Inertial>& interior_pi,
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& interior_phi,

      const Scalar<DataVector>& psi_scalar_interior,
      const Scalar<DataVector>& pi_scalar_interior,
      const tnsr::i<DataVector, 3>& phi_scalar_interior,

      const tnsr::I<DataVector, 3, Frame::Inertial>& /*coords*/,
      const Scalar<DataVector>& interior_gamma1,
      const Scalar<DataVector>& interior_gamma2,
      const Scalar<DataVector>& interior_lapse,
      const tnsr::I<DataVector, 3>& interior_shift,
      const tnsr::II<DataVector, 3>& interior_inv_spatial_metric,
      const tnsr::AA<DataVector, 3,
                     Frame::Inertial>& /*inverse_spacetime_metric*/,
      const tnsr::A<DataVector, 3, Frame::Inertial>&
      /*spacetime_unit_normal_vector*/,
      const tnsr::iaa<DataVector, 3,
                      Frame::Inertial>& /*three_index_constraint*/,
      const tnsr::a<DataVector, 3, Frame::Inertial>& /*gauge_source*/,
      const tnsr::ab<DataVector, 3, Frame::Inertial>&
      /*spacetime_deriv_gauge_source*/,
      const Scalar<DataVector>& interior_gamma1_scalar,
      const Scalar<DataVector>& interior_gamma2_scalar,

      // c.f. dg_interior_dt_vars_tags
      const tnsr::aa<DataVector, 3, Frame::Inertial>&
      /*logical_dt_spacetime_metric*/,
      const tnsr::aa<DataVector, 3, Frame::Inertial>& /*logical_dt_pi*/,
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& /*logical_dt_phi*/,

      const Scalar<DataVector>& /* logical_dt_psi_scalar*/,
      const Scalar<DataVector>& /*logical_dt_pi_scalar*/,
      const tnsr::i<DataVector, 3>& /*logical_dt_phi_scalar*/,

      // c.f. dg_interior_deriv_vars_tags
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& /*d_spacetime_metric*/,
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& /*d_pi*/,
      const tnsr::ijaa<DataVector, 3, Frame::Inertial>& /*d_phi*/,

      const tnsr::i<DataVector, 3, Frame::Inertial>& /*d_psi_scalar*/,
      const tnsr::i<DataVector, 3, Frame::Inertial>& /*d_pi_scalar*/,
      const tnsr::ij<DataVector, 3, Frame::Inertial>& /*d_phi_scalar*/);

  using dg_interior_dt_vars_tags =
      tmpl::list<::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>,
                 ::Tags::dt<gh::Tags::Pi<DataVector, 3>>,
                 ::Tags::dt<gh::Tags::Phi<DataVector, 3>>,
                 ::Tags::dt<CurvedScalarWave::Tags::Psi>,
                 ::Tags::dt<CurvedScalarWave::Tags::Pi>,
                 ::Tags::dt<CurvedScalarWave::Tags::Phi<3>>>;
  using dg_interior_deriv_vars_tags =
      tmpl::list<::Tags::deriv<gr::Tags::SpacetimeMetric<DataVector, 3>,
                               tmpl::size_t<3>, Frame::Inertial>,
                 ::Tags::deriv<gh::Tags::Pi<DataVector, 3>, tmpl::size_t<3>,
                               Frame::Inertial>,
                 ::Tags::deriv<gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                               Frame::Inertial>,
                 ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                               Frame::Inertial>,
                 ::Tags::deriv<CurvedScalarWave::Tags::Pi, tmpl::size_t<3>,
                               Frame::Inertial>,
                 ::Tags::deriv<CurvedScalarWave::Tags::Phi<3>, tmpl::size_t<3>,
                               Frame::Inertial>>;

  std::optional<std::string> dg_time_derivative(
      gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*>
          dt_spacetime_metric_correction,
      gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> dt_pi_correction,
      gsl::not_null<tnsr::iaa<DataVector, 3, Frame::Inertial>*>
          dt_phi_correction,

      gsl::not_null<Scalar<DataVector>*> dt_psi_scalar_correction,
      gsl::not_null<Scalar<DataVector>*> dt_pi_scalar_correction,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          dt_phi_scalar_correction,

      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector,
      // c.f. dg_interior_evolved_variables_tags
      const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
      const tnsr::aa<DataVector, 3, Frame::Inertial>& pi,
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& phi,

      const Scalar<DataVector>& psi_scalar, const Scalar<DataVector>& pi_scalar,
      const tnsr::i<DataVector, 3, Frame::Inertial>& phi_scalar,

      // c.f. dg_interior_temporary_tags
      const tnsr::I<DataVector, 3, Frame::Inertial>& coords,
      const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
      const tnsr::II<DataVector, 3>& /*interior_inv_spatial_metric*/,
      const tnsr::AA<DataVector, 3, Frame::Inertial>& inverse_spacetime_metric,
      const tnsr::A<DataVector, 3, Frame::Inertial>&
          spacetime_unit_normal_vector,
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& three_index_constraint,
      const tnsr::a<DataVector, 3, Frame::Inertial>& gauge_source,
      const tnsr::ab<DataVector, 3, Frame::Inertial>&
          spacetime_deriv_gauge_source,
      const Scalar<DataVector>& gamma1_scalar,
      const Scalar<DataVector>& gamma2_scalar,

      // c.f. dg_interior_dt_vars_tags
      const tnsr::aa<DataVector, 3, Frame::Inertial>&
          logical_dt_spacetime_metric,
      const tnsr::aa<DataVector, 3, Frame::Inertial>& logical_dt_pi,
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& logical_dt_phi,

      const Scalar<DataVector>& logical_dt_psi_scalar,
      const Scalar<DataVector>& logical_dt_pi_scalar,
      const tnsr::i<DataVector, 3>& logical_dt_phi_scalar,

      // c.f. dg_interior_deriv_vars_tags
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& d_spacetime_metric,
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& d_pi,
      const tnsr::ijaa<DataVector, 3, Frame::Inertial>& d_phi,

      const tnsr::i<DataVector, 3, Frame::Inertial>& d_psi_scalar,
      const tnsr::i<DataVector, 3, Frame::Inertial>& d_pi_scalar,
      const tnsr::ij<DataVector, 3, Frame::Inertial>& d_phi_scalar) const;

 private:
  gh::BoundaryConditions::ConstraintPreservingBjorhus<3>
      constraint_preserving_{};
  CurvedScalarWave::BoundaryConditions::ConstraintPreservingSphericalRadiation<
      3>
      csw_constraint_preserving_{};
};
}  // namespace ScalarTensor::BoundaryConditions
