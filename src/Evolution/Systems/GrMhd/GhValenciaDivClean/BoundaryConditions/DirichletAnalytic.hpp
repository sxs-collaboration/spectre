// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DgSubcell/GhostZoneLogicalCoordinates.hpp"
#include "Evolution/DgSubcell/SliceTensor.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean::BoundaryConditions {
/*!
 * \brief Sets Dirichlet boundary conditions using the analytic solution or
 * analytic data.
 *
 * \warning If the analytic prescription is zero density/pressure/energy, then
 * computing the primitives will fail since the specific enthalpy has
 * \f$p/\rho\f$ in it.
 */
class DirichletAnalytic final : public BoundaryCondition {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "DirichletAnalytic boundary conditions using either analytic solution or "
      "analytic data."};

  DirichletAnalytic() = default;
  DirichletAnalytic(DirichletAnalytic&&) = default;
  DirichletAnalytic& operator=(DirichletAnalytic&&) = default;
  DirichletAnalytic(const DirichletAnalytic&) = default;
  DirichletAnalytic& operator=(const DirichletAnalytic&) = default;
  ~DirichletAnalytic() override = default;

  explicit DirichletAnalytic(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, DirichletAnalytic);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>,
                 ::gh::ConstraintDamping::Tags::ConstraintGamma1,
                 ::gh::ConstraintDamping::Tags::ConstraintGamma2>;
  using dg_interior_primitive_variables_tags = tmpl::list<>;
  using dg_gridless_tags = tmpl::list<
      ::Tags::Time, ::Tags::AnalyticSolutionOrData>;

  template <typename AnalyticSolutionOrData>
  std::optional<std::string> dg_ghost(
      const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*>
          spacetime_metric,
      const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> pi,
      const gsl::not_null<tnsr::iaa<DataVector, 3, Frame::Inertial>*> phi,
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

      const gsl::not_null<Scalar<DataVector>*> gamma1,
      const gsl::not_null<Scalar<DataVector>*> gamma2,
      const gsl::not_null<Scalar<DataVector>*> lapse,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
      const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<
          tnsr::I<DataVector, 3, Frame::Inertial>>& /*face_mesh_velocity*/,
      const tnsr::i<DataVector, 3, Frame::Inertial>& /*normal_covector*/,
      const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,
      const tnsr::I<DataVector, 3, Frame::Inertial>& coords,
      const Scalar<DataVector>& interior_gamma1,
      const Scalar<DataVector>& interior_gamma2, const double time,
      const AnalyticSolutionOrData& analytic_solution_or_data) const {
    *gamma1 = interior_gamma1;
    *gamma2 = interior_gamma2;

    auto boundary_values = [&analytic_solution_or_data, &coords, &time]() {
      if constexpr (is_analytic_solution_v<AnalyticSolutionOrData>) {
        return analytic_solution_or_data.variables(
            coords, time,
            tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                       hydro::Tags::ElectronFraction<DataVector>,
                       hydro::Tags::SpecificInternalEnergy<DataVector>,
                       hydro::Tags::SpecificEnthalpy<DataVector>,
                       hydro::Tags::Pressure<DataVector>,
                       hydro::Tags::SpatialVelocity<DataVector, 3>,
                       hydro::Tags::LorentzFactor<DataVector>,
                       hydro::Tags::MagneticField<DataVector, 3>,
                       hydro::Tags::DivergenceCleaningField<DataVector>,
                       gr::Tags::SpatialMetric<DataVector, 3>,
                       gr::Tags::InverseSpatialMetric<DataVector, 3>,
                       gr::Tags::SqrtDetSpatialMetric<DataVector>,
                       gr::Tags::Lapse<DataVector>,
                       gr::Tags::Shift<DataVector, 3>,
                       gr::Tags::SpacetimeMetric<DataVector, 3>,
                       ::gh::Tags::Pi<DataVector, 3>,
                       ::gh::Tags::Phi<DataVector, 3>>{});

      } else {
        (void)time;
        return analytic_solution_or_data.variables(
            coords, tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                               hydro::Tags::ElectronFraction<DataVector>,
                               hydro::Tags::SpecificInternalEnergy<DataVector>,
                               hydro::Tags::SpecificEnthalpy<DataVector>,
                               hydro::Tags::Pressure<DataVector>,
                               hydro::Tags::SpatialVelocity<DataVector, 3>,
                               hydro::Tags::LorentzFactor<DataVector>,
                               hydro::Tags::MagneticField<DataVector, 3>,
                               hydro::Tags::DivergenceCleaningField<DataVector>,
                               gr::Tags::SpatialMetric<DataVector, 3>,
                               gr::Tags::InverseSpatialMetric<DataVector, 3>,
                               gr::Tags::SqrtDetSpatialMetric<DataVector>,
                               gr::Tags::Lapse<DataVector>,
                               gr::Tags::Shift<DataVector, 3>,
                               gr::Tags::SpacetimeMetric<DataVector, 3>,
                               ::gh::Tags::Pi<DataVector, 3>,
                               ::gh::Tags::Phi<DataVector, 3>>{});
      }
    }();

    *spacetime_metric =
        get<gr::Tags::SpacetimeMetric<DataVector, 3>>(boundary_values);
    *pi = get<::gh::Tags::Pi<DataVector, 3>>(boundary_values);
    *phi = get<::gh::Tags::Phi<DataVector, 3>>(boundary_values);
    *lapse = get<gr::Tags::Lapse<DataVector>>(boundary_values);
    *shift = get<gr::Tags::Shift<DataVector, 3>>(boundary_values);
    *inv_spatial_metric =
        get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(boundary_values);

    grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
        tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi,
        get<hydro::Tags::RestMassDensity<DataVector>>(boundary_values),
        get<hydro::Tags::ElectronFraction<DataVector>>(boundary_values),
        get<hydro::Tags::SpecificInternalEnergy<DataVector>>(boundary_values),
        get<hydro::Tags::Pressure<DataVector>>(boundary_values),
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(boundary_values),
        get<hydro::Tags::LorentzFactor<DataVector>>(boundary_values),
        get<hydro::Tags::MagneticField<DataVector, 3>>(boundary_values),
        get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(boundary_values),
        get<gr::Tags::SpatialMetric<DataVector, 3>>(boundary_values),
        get<hydro::Tags::DivergenceCleaningField<DataVector>>(boundary_values));

    grmhd::ValenciaDivClean::ComputeFluxes::apply(
        tilde_d_flux, tilde_ye_flux, tilde_tau_flux, tilde_s_flux, tilde_b_flux,
        tilde_phi_flux, *tilde_d, *tilde_ye, *tilde_tau, *tilde_s, *tilde_b,
        *tilde_phi, *lapse, *shift,
        get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(boundary_values),
        get<gr::Tags::SpatialMetric<DataVector, 3>>(boundary_values),
        get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(boundary_values),
        get<hydro::Tags::Pressure<DataVector>>(boundary_values),
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(boundary_values),
        get<hydro::Tags::LorentzFactor<DataVector>>(boundary_values),
        get<hydro::Tags::MagneticField<DataVector, 3>>(boundary_values));

    return {};
  }

  using fd_interior_evolved_variables_tags = tmpl::list<>;
  using fd_interior_temporary_tags =
      tmpl::list<evolution::dg::subcell::Tags::Mesh<3>>;
  using fd_interior_primitive_variables_tags = tmpl::list<>;
  using fd_gridless_tags =
      tmpl::list<::Tags::Time, ::domain::Tags::FunctionsOfTime,
                 domain::Tags::ElementMap<3, Frame::Grid>,
                 domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                             Frame::Inertial>,
                 fd::Tags::Reconstructor, ::Tags::AnalyticSolutionOrData>;

  template <typename AnalyticSolutionOrData>
  void fd_ghost(
      const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*>
          spacetime_metric,
      const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> pi,
      const gsl::not_null<tnsr::iaa<DataVector, 3, Frame::Inertial>*> phi,
      const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      const gsl::not_null<Scalar<DataVector>*> electron_fraction,
      const gsl::not_null<Scalar<DataVector>*> pressure,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          lorentz_factor_times_spatial_velocity,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          magnetic_field,
      const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
      const Direction<3>& direction,

      // fd_interior_temporary_tags
      const Mesh<3> subcell_mesh,

      // fd_gridless_tags
      const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const ElementMap<3, Frame::Grid>& logical_to_grid_map,
      const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
          grid_to_inertial_map,
      const fd::Reconstructor& reconstructor,
      const AnalyticSolutionOrData& analytic_solution_or_data) const {
    const size_t ghost_zone_size{reconstructor.ghost_zone_size()};

    const auto ghost_logical_coords =
        evolution::dg::subcell::fd::ghost_zone_logical_coordinates(
            subcell_mesh, ghost_zone_size, direction);

    const auto ghost_inertial_coords = grid_to_inertial_map(
        logical_to_grid_map(ghost_logical_coords), time, functions_of_time);

    // Compute FD ghost data with the analytic data or solution
    auto boundary_values = [&analytic_solution_or_data, &ghost_inertial_coords,
                            &time]() {
      if constexpr (std::is_base_of_v<MarkAsAnalyticData,
                                      AnalyticSolutionOrData>) {
        (void)time;
        return analytic_solution_or_data.variables(
            ghost_inertial_coords,
            tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                       hydro::Tags::ElectronFraction<DataVector>,
                       hydro::Tags::Pressure<DataVector>,
                       hydro::Tags::SpatialVelocity<DataVector, 3>,
                       hydro::Tags::LorentzFactor<DataVector>,
                       hydro::Tags::MagneticField<DataVector, 3>,
                       hydro::Tags::DivergenceCleaningField<DataVector>,
                       gr::Tags::SpacetimeMetric<DataVector, 3>,
                       ::gh::Tags::Pi<DataVector, 3>,
                       ::gh::Tags::Phi<DataVector, 3>>{});
      } else {
        return analytic_solution_or_data.variables(
            ghost_inertial_coords, time,
            tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                       hydro::Tags::ElectronFraction<DataVector>,
                       hydro::Tags::Pressure<DataVector>,
                       hydro::Tags::SpatialVelocity<DataVector, 3>,
                       hydro::Tags::LorentzFactor<DataVector>,
                       hydro::Tags::MagneticField<DataVector, 3>,
                       hydro::Tags::DivergenceCleaningField<DataVector>,
                       gr::Tags::SpacetimeMetric<DataVector, 3>,
                       ::gh::Tags::Pi<DataVector, 3>,
                       ::gh::Tags::Phi<DataVector, 3>>{});
      }
    }();

    *spacetime_metric =
        get<gr::Tags::SpacetimeMetric<DataVector, 3>>(boundary_values);
    *pi = get<::gh::Tags::Pi<DataVector, 3>>(boundary_values);
    *phi = get<::gh::Tags::Phi<DataVector, 3>>(boundary_values);
    *rest_mass_density =
        get<hydro::Tags::RestMassDensity<DataVector>>(boundary_values);
    *electron_fraction =
        get<hydro::Tags::ElectronFraction<DataVector>>(boundary_values);
    *pressure = get<hydro::Tags::Pressure<DataVector>>(boundary_values);

    for (size_t i = 0; i < 3; ++i) {
      auto& lorentz_factor =
          get<hydro::Tags::LorentzFactor<DataVector>>(boundary_values);
      auto& spatial_velocity =
          get<hydro::Tags::SpatialVelocity<DataVector, 3>>(boundary_values);
      (*lorentz_factor_times_spatial_velocity).get(i) =
          get(lorentz_factor) * spatial_velocity.get(i);
    }

    *magnetic_field =
        get<hydro::Tags::MagneticField<DataVector, 3>>(boundary_values);
    *divergence_cleaning_field =
        get<hydro::Tags::DivergenceCleaningField<DataVector>>(boundary_values);
  }
};
}  // namespace grmhd::GhValenciaDivClean::BoundaryConditions
