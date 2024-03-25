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
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Bjorhus.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/HydroFreeOutflow.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean::BoundaryConditions {
/*!
 * \brief Sets constraint-preserving boundary conditions on the spacetime
 * variables and hydro free outflow on the GRMHD variables.
 *
 * \warning This is only implemented for DG, on FD you get an error. The
 * reason is that some care and experimentation is necessary to impose the
 * boundary condition correctly on FD.
 */
class ConstraintPreservingFreeOutflow final : public BoundaryCondition {
 public:
  using options =
      typename gh::BoundaryConditions::ConstraintPreservingBjorhus<3>::options;
  static constexpr Options::String help{
      "ConstraintPreservingAnalytic boundary conditions  for GH variables and "
      "hydro free outflow for GRMHD."};

  ConstraintPreservingFreeOutflow() = default;
  explicit ConstraintPreservingFreeOutflow(
      gh::BoundaryConditions::detail::ConstraintPreservingBjorhusType type);

  ConstraintPreservingFreeOutflow(ConstraintPreservingFreeOutflow&&) = default;
  ConstraintPreservingFreeOutflow& operator=(
      ConstraintPreservingFreeOutflow&&) = default;
  ConstraintPreservingFreeOutflow(const ConstraintPreservingFreeOutflow&) =
      default;
  ConstraintPreservingFreeOutflow& operator=(
      const ConstraintPreservingFreeOutflow&) = default;
  ~ConstraintPreservingFreeOutflow() override = default;

  explicit ConstraintPreservingFreeOutflow(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition,
      ConstraintPreservingFreeOutflow);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::GhostAndTimeDerivative;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags =
      tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
                 gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>>;
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
                 gh::Tags::SpacetimeDerivGaugeH<DataVector, 3>>;
  using dg_interior_primitive_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::Temperature<DataVector>>;
  using dg_gridless_tags = tmpl::list<>;

  static std::optional<std::string> dg_ghost(
      gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> spacetime_metric,
      gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> pi,
      gsl::not_null<tnsr::iaa<DataVector, 3, Frame::Inertial>*> phi,
      gsl::not_null<Scalar<DataVector>*> tilde_d,
      gsl::not_null<Scalar<DataVector>*> tilde_ye,
      gsl::not_null<Scalar<DataVector>*> tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
      gsl::not_null<Scalar<DataVector>*> tilde_phi,

      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_d_flux,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_ye_flux,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_tau_flux,
      gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> tilde_s_flux,
      gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_phi_flux,

      gsl::not_null<Scalar<DataVector>*> gamma1,
      gsl::not_null<Scalar<DataVector>*> gamma2,
      gsl::not_null<Scalar<DataVector>*> lapse,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          spatial_velocity_one_form,
      gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      gsl::not_null<Scalar<DataVector>*> electron_fraction,
      gsl::not_null<Scalar<DataVector>*> temperature,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> spatial_velocity,
      gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector,

      const tnsr::aa<DataVector, 3, Frame::Inertial>& interior_spacetime_metric,
      const tnsr::aa<DataVector, 3, Frame::Inertial>& interior_pi,
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& interior_phi,

      const Scalar<DataVector>& interior_rest_mass_density,
      const Scalar<DataVector>& interior_electron_fraction,
      const Scalar<DataVector>& interior_specific_internal_energy,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
      const Scalar<DataVector>& interior_lorentz_factor,
      const Scalar<DataVector>& interior_pressure,
      const Scalar<DataVector>& interior_temperature,

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

      // c.f. dg_interior_dt_vars_tags
      const tnsr::aa<DataVector, 3, Frame::Inertial>&
      /*logical_dt_spacetime_metric*/,
      const tnsr::aa<DataVector, 3, Frame::Inertial>& /*logical_dt_pi*/,
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& /*logical_dt_phi*/,
      // c.f. dg_interior_deriv_vars_tags
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& /*d_spacetime_metric*/,
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& /*d_pi*/,
      const tnsr::ijaa<DataVector, 3, Frame::Inertial>& /*d_phi*/);

  using dg_interior_dt_vars_tags =
      tmpl::list<::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>,
                 ::Tags::dt<gh::Tags::Pi<DataVector, 3>>,
                 ::Tags::dt<gh::Tags::Phi<DataVector, 3>>>;
  using dg_interior_deriv_vars_tags =
      tmpl::list<::Tags::deriv<gr::Tags::SpacetimeMetric<DataVector, 3>,
                               tmpl::size_t<3>, Frame::Inertial>,
                 ::Tags::deriv<gh::Tags::Pi<DataVector, 3>, tmpl::size_t<3>,
                               Frame::Inertial>,
                 ::Tags::deriv<gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                               Frame::Inertial>>;

  std::optional<std::string> dg_time_derivative(
      gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*>
          dt_spacetime_metric_correction,
      gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> dt_pi_correction,
      gsl::not_null<tnsr::iaa<DataVector, 3, Frame::Inertial>*>
          dt_phi_correction,
      gsl::not_null<Scalar<DataVector>*> dt_tilde_d,
      gsl::not_null<Scalar<DataVector>*> dt_tilde_ye,
      gsl::not_null<Scalar<DataVector>*> dt_tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> dt_tilde_s,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> dt_tilde_b,
      gsl::not_null<Scalar<DataVector>*> dt_tilde_phi,

      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector,
      // c.f. dg_interior_evolved_variables_tags
      const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
      const tnsr::aa<DataVector, 3, Frame::Inertial>& pi,
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& phi,
      // c.f. dg_interior_primitive_variables_tags
      const Scalar<DataVector>& /*interior_rest_mass_density*/,
      const Scalar<DataVector>& /*interior_electron_fraction*/,
      const Scalar<DataVector>& /*interior_specific_internal_energy*/,
      const tnsr::I<DataVector, 3,
                    Frame::Inertial>& /*interior_spatial_velocity*/,
      const tnsr::I<DataVector, 3,
                    Frame::Inertial>& /*interior_magnetic_field*/,
      const Scalar<DataVector>& /*interior_lorentz_factor*/,
      const Scalar<DataVector>& /*interior_pressure*/,
      const Scalar<DataVector>& /*interior_temperature*/,

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
      // c.f. dg_interior_dt_vars_tags
      const tnsr::aa<DataVector, 3, Frame::Inertial>&
          logical_dt_spacetime_metric,
      const tnsr::aa<DataVector, 3, Frame::Inertial>& logical_dt_pi,
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& logical_dt_phi,
      // c.f. dg_interior_deriv_vars_tags
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& d_spacetime_metric,
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& d_pi,
      const tnsr::ijaa<DataVector, 3, Frame::Inertial>& d_phi) const;

  using fd_interior_evolved_variables_tags = tmpl::list<>;
  using fd_interior_temporary_tags = tmpl::list<>;
  using fd_interior_primitive_variables_tags = tmpl::list<>;
  using fd_gridless_tags = tmpl::list<>;

  [[noreturn]] static void fd_ghost(
      const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*>
      /*spacetime_metric*/,
      const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> /*pi*/,
      const gsl::not_null<tnsr::iaa<DataVector, 3, Frame::Inertial>*> /*phi*/,
      const gsl::not_null<Scalar<DataVector>*> /*rest_mass_density*/,
      const gsl::not_null<Scalar<DataVector>*> /*electron_fraction*/,
      const gsl::not_null<Scalar<DataVector>*> /*pressure*/,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
      /*lorentz_factor_times_spatial_velocity*/,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
      /*magnetic_field*/,
      const gsl::not_null<Scalar<DataVector>*> /*divergence_cleaning_field*/,
      const Direction<3>& /*direction*/) {
    ERROR(
        "Not implemented because it's not trivial to figure out what the right "
        "way of handling this case is.");
  }

 private:
  gh::BoundaryConditions::ConstraintPreservingBjorhus<3>
      constraint_preserving_{};
};
}  // namespace grmhd::GhValenciaDivClean::BoundaryConditions
