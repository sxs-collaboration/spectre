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
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/Hydro/Temperature.hpp"
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
 * \brief Apply "soft" reflective boundary condition as described in
 * \cite Shiokawa2011ih.
 *
 * All primitive variables at the boundary are copied into ghost zone except :
 *
 *  - If \f$n_iv^i \leq 0\f$ where \f$v^i\f$ is spatial velocity and \f$n_i\f$
 * is outward directed normal covector, copy the values of \f$v^i\f$ at the
 * boundary to ghost zone. If \f$n_iv^i>0\f$, spatial velocity in the ghost zone
 * is modified such that the sign of normal component is inverted at the
 * interface i.e. \f$v_\text{ghost}^i = v^i - 2*(n_jv^j)n^i\f$.
 *
 *  - If \f$n_iB^i \leq 0\f$ where \f$B^i\f$ is magnetic field and \f$n_i\f$
 * is outward directed normal covector, copy the values of \f$B^i\f$ at the
 * boundary to ghost zone. If \f$n_iB^i>0\f$, magnetic field in the ghost zone
 * is modified such that the sign of normal component is inverted at the
 * interface i.e. \f$B_\text{ghost}^i = B^i - 2*(n_jB^j)n^i\f$.
 *
 *  - If reflect_both is true, then spatial velocity and magnetic field are
 * are inverted regardless of whether the normal component is pointing outward
 * or inward.
 *
 *  - However, regardless of whether the normal component of the spatial
 * velocity $n_iv^i$ is pointing outward or inward, the lorentz factor \f$W\f$
 * is copied into ghost zone without any modification.
 *
 *  - Divergence cleaning scalar field \f$\Phi\f$ is set to zero in ghost zone.
 *
 */
class Reflective final : public BoundaryCondition {
 private:
  bool reflect_both_{false};

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

  using prim_tags_for_reconstruction =
      tmpl::list<RestMassDensity, ElectronFraction, Temperature,
                 LorentzFactorTimesSpatialVelocity, MagneticField,
                 DivergenceCleaningField>;

  template <typename T>
  using Flux = ::Tags::Flux<T, tmpl::size_t<3>, Frame::Inertial>;

 public:
  struct ReflectBoth {
    using type = bool;
    static constexpr Options::String help = {
        "Reflect both outgoing and incoming normal component of "
        "spatial velocity and magnetic field."};
  };
  using options = tmpl::list<ReflectBoth>;
  static constexpr Options::String help{
      "Reflective boundary conditions, inverting the sign "
      "of outgoing normal component of spatial velocity "
      "and magnetic field."};

  Reflective() = default;
  Reflective(Reflective&&) = default;
  Reflective& operator=(Reflective&&) = default;
  Reflective(const Reflective&) = default;
  Reflective& operator=(const Reflective&) = default;
  ~Reflective() override = default;

  explicit Reflective(bool reflect_both);

  explicit Reflective(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, Reflective);

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
                 hydro::Tags::Pressure<DataVector>>;
  using dg_interior_temporary_tags = tmpl::list<Shift, Lapse, InvSpatialMetric>;
  using dg_gridless_tags = tmpl::list<>;

  std::optional<std::string> dg_ghost(
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

      gsl::not_null<Scalar<DataVector>*> lapse,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
      gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
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

      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
      const Scalar<DataVector>& interior_lapse,
      const tnsr::II<DataVector, 3, Frame::Inertial>&
          interior_inv_spatial_metric) const;

  using fd_interior_evolved_variables_tags = tmpl::list<>;
  using fd_interior_temporary_tags =
      tmpl::list<evolution::dg::subcell::Tags::Mesh<3>, Shift, Lapse,
                 SpatialMetric>;
  using fd_interior_primitive_variables_tags =
      tmpl::list<RestMassDensity, ElectronFraction, Temperature,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>, MagneticField>;

  using fd_gridless_tags = tmpl::list<fd::Tags::Reconstructor>;

  void fd_ghost(
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
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,

      // gridless tags
      const fd::Reconstructor& reconstructor) const;

  // have an impl to make sharing code with GH+GRMHD easy
  void fd_ghost_impl(
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
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_spatial_metric,
      const Scalar<DataVector>& interior_lapse,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,

      size_t ghost_zone_size, bool need_tags_for_fluxes) const;
  };
}  // namespace grmhd::ValenciaDivClean::BoundaryConditions
