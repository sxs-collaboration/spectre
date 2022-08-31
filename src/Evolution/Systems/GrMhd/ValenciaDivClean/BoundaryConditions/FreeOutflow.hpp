// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace grmhd::ValenciaDivClean::BoundaryConditions {
/*!
 * \brief Apply hydrodynamic free outflow and no inflow boundary condition to
 * GRMHD primitive variables.
 *
 * All primitive variables at the boundary are copied into ghost zone except :
 *
 *  - If \f$n_iv^i \geq 0\f$ where \f$v^i\f$ is spatial velocity and \f$n_i\f$
 * is outward directed normal covector, copy the values of \f$v^i\f$ at the
 * boundary to ghost zone. If \f$n_iv^i<0\f$, spatial velocity in the ghost zone
 * is modified such that the normal component is zero at the interface i.e.
 * \f$v_\text{ghost}^i = v^i - (n_jv^j)n^i\f$.
 *
 *  - However, regardless of whether the normal component of the spatial
 * velocity $n_iv^i$ is pointing outward or inward, the lorentz factor \f$W\f$
 * is copied into ghost zone without any modification.
 *
 *  - Divergence cleaning scalar field \f$\Phi\f$ is set to zero in ghost zone.
 *
 */
class FreeOutflow final : public BoundaryCondition {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Free outflow & no inflow boundary condition on GRMHD primitive "
      "variables"};

  FreeOutflow() = default;
  FreeOutflow(FreeOutflow&&) = default;
  FreeOutflow& operator=(FreeOutflow&&) = default;
  FreeOutflow(const FreeOutflow&) = default;
  FreeOutflow& operator=(const FreeOutflow&) = default;
  ~FreeOutflow() override = default;

  explicit FreeOutflow(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, FreeOutflow);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_primitive_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::SpecificEnthalpy<DataVector>>;
  using dg_interior_temporary_tags = tmpl::list<
      gr::Tags::Shift<3, Frame::Inertial, DataVector>,
      gr::Tags::Lapse<DataVector>,
      gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>>;
  using dg_gridless_tags = tmpl::list<>;

  static std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> tilde_d,
      const gsl::not_null<Scalar<DataVector>*> tilde_tau,
      const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
      const gsl::not_null<Scalar<DataVector>*> tilde_phi,

      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          tilde_d_flux,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          tilde_tau_flux,
      const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*>
          tilde_s_flux,
      const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>
          tilde_b_flux,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          tilde_phi_flux,

      const gsl::not_null<Scalar<DataVector>*> lapse,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
      const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
      /*face_mesh_velocity*/,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
          outward_directed_normal_vector,

      const Scalar<DataVector>& interior_rest_mass_density,
      const Scalar<DataVector>& interior_specific_internal_energy,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
      const Scalar<DataVector>& interior_lorentz_factor,
      const Scalar<DataVector>& interior_pressure,
      const Scalar<DataVector>& interior_specific_enthalpy,

      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
      const Scalar<DataVector>& interior_lapse,
      const tnsr::II<DataVector, 3, Frame::Inertial>&
          interior_inv_spatial_metric);
};
}  // namespace grmhd::ValenciaDivClean::BoundaryConditions
