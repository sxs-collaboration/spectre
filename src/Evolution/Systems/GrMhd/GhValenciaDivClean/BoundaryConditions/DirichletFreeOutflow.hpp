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
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/HydroFreeOutflow.hpp"
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
 * analytic data on the spacetime variables and hydro free outflow on the GRMHD
 * variables.
 */
class DirichletFreeOutflow final : public BoundaryCondition {
 private:
  template <typename T>
  using Flux = ::Tags::Flux<T, tmpl::size_t<3>, Frame::Inertial>;

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "DirichletAnalytic boundary conditions using either analytic solution or "
      "analytic data for GH variables and hydro free outflow for GRMHD."};

  DirichletFreeOutflow() = default;
  DirichletFreeOutflow(DirichletFreeOutflow&&) = default;
  DirichletFreeOutflow& operator=(DirichletFreeOutflow&&) = default;
  DirichletFreeOutflow(const DirichletFreeOutflow&) = default;
  DirichletFreeOutflow& operator=(const DirichletFreeOutflow&) = default;
  ~DirichletFreeOutflow() override = default;

  explicit DirichletFreeOutflow(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, DirichletFreeOutflow);

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
  using dg_interior_primitive_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::Pressure<DataVector>>;
  using dg_gridless_tags =
      tmpl::list<::Tags::Time, ::Tags::AnalyticSolutionOrData>;

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

      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector,

      const Scalar<DataVector>& interior_rest_mass_density,
      const Scalar<DataVector>& interior_electron_fraction,
      const Scalar<DataVector>& interior_specific_internal_energy,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
      const Scalar<DataVector>& interior_lorentz_factor,
      const Scalar<DataVector>& interior_pressure,

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
            tmpl::list<gr::Tags::SpatialMetric<DataVector, 3>,
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
            coords, tmpl::list<gr::Tags::SpatialMetric<DataVector, 3>,
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

    return grmhd::ValenciaDivClean::BoundaryConditions::HydroFreeOutflow::
        dg_ghost(tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi,

                 tilde_d_flux, tilde_ye_flux, tilde_tau_flux, tilde_s_flux,
                 tilde_b_flux, tilde_phi_flux,

                 lapse, shift, inv_spatial_metric,

                 face_mesh_velocity, normal_covector, normal_vector,

                 interior_rest_mass_density, interior_electron_fraction,
                 interior_specific_internal_energy, interior_spatial_velocity,
                 interior_magnetic_field, interior_lorentz_factor,
                 interior_pressure,

                 *shift, *lapse, *inv_spatial_metric);
  }

  using fd_interior_evolved_variables_tags = tmpl::list<>;
  using fd_interior_temporary_tags =
      tmpl::list<evolution::dg::subcell::Tags::Mesh<3>>;
  using fd_interior_primitive_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>>;
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

      // interior prim vars tags
      const Scalar<DataVector>& interior_rest_mass_density,
      const Scalar<DataVector>& interior_electron_fraction,
      const Scalar<DataVector>& interior_pressure,
      const Scalar<DataVector>& interior_specific_internal_energy,
      const Scalar<DataVector>& interior_lorentz_factor,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,

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
            tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
                       ::gh::Tags::Pi<DataVector, 3>,
                       ::gh::Tags::Phi<DataVector, 3>>{});
      } else {
        return analytic_solution_or_data.variables(
            ghost_inertial_coords, time,
            tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
                       ::gh::Tags::Pi<DataVector, 3>,
                       ::gh::Tags::Phi<DataVector, 3>>{});
      }
    }();

    *spacetime_metric =
        get<gr::Tags::SpacetimeMetric<DataVector, 3>>(boundary_values);
    *pi = get<::gh::Tags::Pi<DataVector, 3>>(boundary_values);
    *phi = get<::gh::Tags::Phi<DataVector, 3>>(boundary_values);

    // Note: Once we support high-order fluxes with GHMHD we will need to
    // handle this correctly.
    std::optional<Variables<db::wrap_tags_in<
        Flux, typename grmhd::ValenciaDivClean::System::flux_variables>>>
        cell_centered_ghost_fluxes{std::nullopt};
    // Set to zero since it shouldn't be used
    Scalar<DataVector> specific_internal_energy{};
    tnsr::I<DataVector, 3> spatial_velocity{};
    Scalar<DataVector> lorentz_factor{};
    const tnsr::I<DataVector, 3> interior_shift{};
    const Scalar<DataVector> interior_lapse{};
    const tnsr::ii<DataVector, 3> interior_spatial_metric{};
    tnsr::ii<DataVector, 3> spatial_metric{};
    tnsr::II<DataVector, 3> inv_spatial_metric{};
    Scalar<DataVector> sqrt_det_spatial_metric{};
    Scalar<DataVector> lapse{};
    tnsr::I<DataVector, 3> shift{};

    grmhd::ValenciaDivClean::BoundaryConditions::HydroFreeOutflow::
        fd_ghost_impl(
            rest_mass_density, electron_fraction, pressure,
            make_not_null(&specific_internal_energy),
            lorentz_factor_times_spatial_velocity,
            make_not_null(&spatial_velocity), make_not_null(&lorentz_factor),
            magnetic_field, divergence_cleaning_field,

            make_not_null(&spatial_metric), make_not_null(&inv_spatial_metric),
            make_not_null(&sqrt_det_spatial_metric), make_not_null(&lapse),
            make_not_null(&shift),

            direction,

            // fd_interior_temporary_tags
            subcell_mesh,

            // fd_interior_primitive_variables_tags
            interior_rest_mass_density, interior_electron_fraction,
            interior_pressure, interior_specific_internal_energy,
            interior_lorentz_factor, interior_spatial_velocity,
            interior_magnetic_field,
            // Note: metric vars are empty because they shouldn't be used
            interior_spatial_metric, interior_lapse, interior_shift,

            // fd_gridless_tags
            reconstructor.ghost_zone_size(),
            cell_centered_ghost_fluxes.has_value());
  }
};
}  // namespace grmhd::GhValenciaDivClean::BoundaryConditions
