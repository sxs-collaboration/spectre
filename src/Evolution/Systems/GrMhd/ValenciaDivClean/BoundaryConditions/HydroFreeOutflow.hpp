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
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t VolumeDim>
class Direction;
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
 * \note In principle we need to recompute the Lorentz factor
 * \f$W_\text{ghost}\f$ using $v_\text{ghost}^i$. However, the case in which
 * skipping that procedure becomes problematic is only when the fluid has
 * relativistic incoming normal speed on the external boundary, which already
 * implies the failure of adopting outflow type boundary conditions.
 *
 *  - Divergence cleaning scalar field \f$\Phi\f$ is set to zero in ghost zone.
 *
 */
class HydroFreeOutflow final : public BoundaryCondition {
 private:
  using RestMassDensity = hydro::Tags::RestMassDensity<DataVector>;
  using ElectronFraction = hydro::Tags::ElectronFraction<DataVector>;
  using Pressure = hydro::Tags::Pressure<DataVector>;
  using LorentzFactorTimesSpatialVelocity =
      hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>;
  using MagneticField = hydro::Tags::MagneticField<DataVector, 3>;
  using DivergenceCleaningField =
      hydro::Tags::DivergenceCleaningField<DataVector>;

  using prim_tags_for_reconstruction =
      tmpl::list<RestMassDensity, ElectronFraction, Pressure,
                 LorentzFactorTimesSpatialVelocity, MagneticField,
                 DivergenceCleaningField>;

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Free outflow & no inflow boundary condition on GRMHD primitive "
      "variables"};

  HydroFreeOutflow() = default;
  HydroFreeOutflow(HydroFreeOutflow&&) = default;
  HydroFreeOutflow& operator=(HydroFreeOutflow&&) = default;
  HydroFreeOutflow(const HydroFreeOutflow&) = default;
  HydroFreeOutflow& operator=(const HydroFreeOutflow&) = default;
  ~HydroFreeOutflow() override = default;

  explicit HydroFreeOutflow(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, HydroFreeOutflow);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_primitive_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
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
      const gsl::not_null<Scalar<DataVector>*> tilde_ye,
      const gsl::not_null<Scalar<DataVector>*> tilde_tau,
      const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
      const gsl::not_null<Scalar<DataVector>*> tilde_phi,

      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          tilde_d_flux,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          tilde_ye_flux,
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
      const Scalar<DataVector>& interior_electron_fraction,
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

  using fd_interior_evolved_variables_tags = tmpl::list<>;
  using fd_interior_temporary_tags =
      tmpl::list<evolution::dg::subcell::Tags::Mesh<3>>;
  using fd_interior_primitive_variables_tags =
      tmpl::list<RestMassDensity, ElectronFraction, Pressure,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>, MagneticField>;
  using fd_gridless_tags = tmpl::list<fd::Tags::Reconstructor>;

  static void fd_ghost(
      gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      gsl::not_null<Scalar<DataVector>*> electron_fraction,
      gsl::not_null<Scalar<DataVector>*> pressure,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          lorentz_factor_times_spatial_velocity,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> magnetic_field,
      gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,

      const Direction<3>& direction,

      // interior temporary tags
      const Mesh<3>& subcell_mesh,

      // interior prim vars tags
      const Scalar<DataVector>& interior_rest_mass_density,
      const Scalar<DataVector>& interior_electron_fraction,
      const Scalar<DataVector>& interior_pressure,
      const Scalar<DataVector>& interior_lorentz_factor,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,

      // gridless tags
      const fd::Reconstructor& reconstructor);

  // have an impl to make sharing code with GH+GRMHD easy
  static void fd_ghost_impl(
      gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      gsl::not_null<Scalar<DataVector>*> electron_fraction,
      gsl::not_null<Scalar<DataVector>*> pressure,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          lorentz_factor_times_spatial_velocity,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> magnetic_field,
      gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,

      const Direction<3>& direction,

      // fd_interior_temporary_tags
      const Mesh<3>& subcell_mesh,

      // fd_interior_primitive_variables_tags
      const Scalar<DataVector>& interior_rest_mass_density,
      const Scalar<DataVector>& interior_electron_fraction,
      const Scalar<DataVector>& interior_pressure,
      const Scalar<DataVector>& interior_lorentz_factor,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,

      size_t ghost_zone_size);
};
}  // namespace grmhd::ValenciaDivClean::BoundaryConditions
