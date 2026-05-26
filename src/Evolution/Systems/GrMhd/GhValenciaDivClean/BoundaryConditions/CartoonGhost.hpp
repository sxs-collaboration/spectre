// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/Cartoon.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Options/String.hpp"
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

namespace grmhd::GhValenciaDivClean::BoundaryConditions {
/*!
 * \brief Apply parity-respecting FD ghost values for internal boundaries in a
 * Cartoon evolution.
 *
 * Domains using the cartoon method have boundaries at $x = 0$, which require
 * ghost data for FD evolution. We fill this FD ghost data by reflecting the
 * data appropriately: due to the symmetry of cartoon simulations, we can
 * determine the parity of each component of an arbitrary tensor, allowing us
 * to either reflect the data or reflect and negate the data.
 *
 * This only has `fd_ghost()` implemented because this boundary uses ZernikeB1
 * bases in DG elements which do not require a boundary condition.
 *
 * Sets GH variables based on their parity, and calls the
 * `ValenciaDivClean::BoundaryConditions::CartoonGhost` implementation to set
 * the hydro variables.
 */
template <typename System>
class CartoonGhost final : public BoundaryCondition,
                           public domain::BoundaryConditions::MarkAsCartoon {
 private:
  using SpacetimeMetric = gr::Tags::SpacetimeMetric<DataVector, 3>;
  using Pi = gh::Tags::Pi<DataVector, 3>;
  using Phi = gh::Tags::Phi<DataVector, 3>;

 public:
  static constexpr bool factory_creatable = false;

  CartoonGhost() = default;
  CartoonGhost(CartoonGhost&&) = default;
  CartoonGhost& operator=(CartoonGhost&&) = default;
  CartoonGhost(const CartoonGhost&) = default;
  CartoonGhost& operator=(const CartoonGhost&) = default;
  ~CartoonGhost() override = default;

  explicit CartoonGhost(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, CartoonGhost);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_primitive_variables_tags = tmpl::list<>;
  using dg_interior_temporary_tags = tmpl::list<>;
  using dg_gridless_tags = tmpl::list<>;

  [[noreturn]] static std::optional<std::string> dg_ghost(
      gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*>
      /*spacetime_metric*/,
      gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> /*pi*/,
      gsl::not_null<tnsr::iaa<DataVector, 3, Frame::Inertial>*> /*phi*/,
      gsl::not_null<Scalar<DataVector>*> /*tilde_d*/,
      gsl::not_null<Scalar<DataVector>*> /*tilde_ye*/,
      gsl::not_null<Scalar<DataVector>*> /*tilde_tau*/,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> /*tilde_s*/,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> /*tilde_b*/,
      gsl::not_null<Scalar<DataVector>*> /*tilde_phi*/,

      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
      /*tilde_d_flux*/,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
      /*tilde_ye_flux*/,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
      /*tilde_tau_flux*/,
      gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*>
      /*tilde_s_flux*/,
      gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>
      /*tilde_b_flux*/,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
      /*tilde_phi_flux*/,

      gsl::not_null<Scalar<DataVector>*> /*gamma1*/,
      gsl::not_null<Scalar<DataVector>*> /*gamma2*/,
      gsl::not_null<Scalar<DataVector>*> /*lapse*/,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> /*shift*/,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
      /*spatial_velocity_one_form*/,
      gsl::not_null<Scalar<DataVector>*> /*rest_mass_density*/,
      gsl::not_null<Scalar<DataVector>*> /*electron_fraction*/,
      gsl::not_null<Scalar<DataVector>*> /*temperature*/,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
      /*spatial_velocity*/,
      gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
      /*inv_spatial_metric*/,

      const std::optional<
          tnsr::I<DataVector, 3, Frame::Inertial>>& /*face_mesh_velocity*/,
      const tnsr::i<DataVector, 3, Frame::Inertial>& /*normal_covector*/,
      const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/) {
    ERROR(
        "dg_ghost() should never be called: the boundary it would apply to "
        "uses ZernikeB1 bases in DG evolution which do not need a boundary "
        "condition. When applying DG boundary conditions this class should be "
        "skipped as it is marked as Cartoon");
  }

  using fd_interior_evolved_variables_tags =
      tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
                 gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>>;
  using fd_interior_temporary_tags =
      tmpl::list<evolution::dg::subcell::Tags::Mesh<3>>;
  using fd_interior_primitive_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::Temperature<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::DivergenceCleaningField<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>>;
  using fd_gridless_tags = tmpl::list<fd::Tags::Reconstructor<System>>;

  static void fd_ghost(
      gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> spacetime_metric,
      gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> pi,
      gsl::not_null<tnsr::iaa<DataVector, 3, Frame::Inertial>*> phi,
      gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      gsl::not_null<Scalar<DataVector>*> electron_fraction,
      gsl::not_null<Scalar<DataVector>*> temperature,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          lorentz_factor_times_spatial_velocity,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> magnetic_field,
      gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
      const Direction<3>& direction,

      // fd_interior_evolved_variables_tags
      const tnsr::aa<DataVector, 3, Frame::Inertial>& interior_spacetime_metric,
      const tnsr::aa<DataVector, 3, Frame::Inertial>& interior_pi,
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& interior_phi,

      // fd_interior_temporary_tags
      const Mesh<3>& subcell_mesh,

      // fd_interior_primitive_variables_tags
      const Scalar<DataVector>& interior_rest_mass_density,
      const Scalar<DataVector>& interior_electron_fraction,
      const Scalar<DataVector>& interior_temperature,
      const Scalar<DataVector>& interior_pressure,
      const Scalar<DataVector>& interior_specific_internal_energy,
      const Scalar<DataVector>& interior_lorentz_factor,
      const Scalar<DataVector>& interior_divergence_cleaning_field,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,

      // fd_gridless_tags
      const fd::Reconstructor<System>& reconstructor);

 private:
  static void fd_ghost_gh_impl(
      gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> spacetime_metric,
      gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> pi,
      gsl::not_null<tnsr::iaa<DataVector, 3, Frame::Inertial>*> phi,
      const Direction<3>& direction,
      const tnsr::aa<DataVector, 3, Frame::Inertial>& interior_spacetime_metric,
      const tnsr::aa<DataVector, 3, Frame::Inertial>& interior_pi,
      const tnsr::iaa<DataVector, 3, Frame::Inertial>& interior_phi,
      const Mesh<3>& subcell_mesh, size_t ghost_zone_size);
};
}  // namespace grmhd::GhValenciaDivClean::BoundaryConditions
