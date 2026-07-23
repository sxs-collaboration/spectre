// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/Cartoon.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
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
 */
class CartoonGhost final : public BoundaryCondition,
                           public domain::BoundaryConditions::MarkAsCartoon {
 private:
  using RestMassDensity = hydro::Tags::RestMassDensity<DataVector>;
  using ElectronFraction = hydro::Tags::ElectronFraction<DataVector>;
  using Temperature = hydro::Tags::Temperature<DataVector>;
  using Pressure = hydro::Tags::Pressure<DataVector>;
  using LorentzFactorTimesSpatialVelocity =
      hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>;
  using MagneticField = hydro::Tags::MagneticField<DataVector, 3>;
  using DivergenceCleaningField =
      hydro::Tags::DivergenceCleaningField<DataVector>;
  using SpecificInternalEnergy =
      hydro::Tags::SpecificInternalEnergy<DataVector>;
  using SpatialVelocity = hydro::Tags::SpatialVelocity<DataVector, 3>;
  using LorentzFactor = hydro::Tags::LorentzFactor<DataVector>;
  using SqrtDetSpatialMetric = gr::Tags::SqrtDetSpatialMetric<DataVector>;
  using SpatialMetric = gr::Tags::SpatialMetric<DataVector, 3>;
  using InvSpatialMetric = gr::Tags::InverseSpatialMetric<DataVector, 3>;
  using Lapse = gr::Tags::Lapse<DataVector>;
  using Shift = gr::Tags::Shift<DataVector, 3>;

  template <typename T>
  using Flux = ::Tags::Flux<T, tmpl::size_t<3>, Frame::Inertial>;

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
      gsl::not_null<Scalar<DataVector>*> /*tilde_d*/,
      gsl::not_null<Scalar<DataVector>*> /*tilde_ye*/,
      gsl::not_null<Scalar<DataVector>*> /*tilde_tau*/,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> /*tilde_s*/,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> /*tilde_b*/,
      gsl::not_null<Scalar<DataVector>*> /*tilde_phi*/,

      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> /*tilde_d_flux*/,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> /*tilde_ye_flux*/,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
      /*tilde_tau_flux*/,
      gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> /*tilde_s_flux*/,
      gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> /*tilde_b_flux*/,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
      /*tilde_phi_flux*/,

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

  using fd_interior_evolved_variables_tags = tmpl::list<>;
  using fd_interior_temporary_tags =
      tmpl::list<evolution::dg::subcell::Tags::Mesh<3>, Shift, Lapse,
                 SpatialMetric>;
  using fd_interior_primitive_variables_tags =
      tmpl::list<RestMassDensity, ElectronFraction, Temperature,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::DivergenceCleaningField<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>, MagneticField>;
  using fd_gridless_tags = tmpl::list<fd::Tags::Reconstructor>;

  static void fd_ghost(
      gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      gsl::not_null<Scalar<DataVector>*> electron_fraction,
      gsl::not_null<Scalar<DataVector>*> temperature,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          lorentz_factor_times_spatial_velocity,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> magnetic_field,
      gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,

      gsl::not_null<std::optional<Variables<db::wrap_tags_in<
          Flux, typename grmhd::ValenciaDivClean::System::flux_variables>>>*>
          cell_centered_ghost_fluxes,

      const Direction<3>& direction,

      // interior temporary tags
      const Mesh<3>& subcell_mesh,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
      const Scalar<DataVector>& interior_lapse,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_spatial_metric,

      // interior prim vars tags
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
      const fd::Reconstructor& reconstructor);

  // have an impl to make sharing code with GH+GRMHD easy
  static void fd_ghost_impl(
      gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      gsl::not_null<Scalar<DataVector>*> electron_fraction,
      gsl::not_null<Scalar<DataVector>*> temperature,
      gsl::not_null<Scalar<DataVector>*> pressure,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          lorentz_factor_times_spatial_velocity,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> spatial_velocity,
      gsl::not_null<Scalar<DataVector>*> lorentz_factor,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> magnetic_field,
      gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*> spatial_metric,
      gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
          inv_spatial_metric,
      gsl::not_null<Scalar<DataVector>*> sqrt_det_spatial_metric,
      gsl::not_null<Scalar<DataVector>*> lapse,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,

      const Direction<3>& direction,

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
      const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_spatial_metric,
      const Scalar<DataVector>& interior_lapse,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,

      size_t ghost_zone_size, bool need_tags_for_fluxes);
};
}  // namespace grmhd::ValenciaDivClean::BoundaryConditions
