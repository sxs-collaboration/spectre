// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/DirichletAnalytic.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
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
#include "Evolution/Systems/GrMhd/ValenciaDivClean/AllSolutions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/Hydro/Temperature.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::BoundaryConditions {
DirichletAnalytic::DirichletAnalytic(const DirichletAnalytic& rhs)
    : BoundaryCondition{dynamic_cast<const BoundaryCondition&>(rhs)},
      analytic_prescription_(rhs.analytic_prescription_->get_clone()) {}

DirichletAnalytic& DirichletAnalytic::operator=(const DirichletAnalytic& rhs) {
  if (&rhs == this) {
    return *this;
  }
  analytic_prescription_ = rhs.analytic_prescription_->get_clone();
  return *this;
}

DirichletAnalytic::DirichletAnalytic(
    std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription)
    : analytic_prescription_(std::move(analytic_prescription)) {}

DirichletAnalytic::DirichletAnalytic(CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DirichletAnalytic::get_clone() const {
  return std::make_unique<DirichletAnalytic>(*this);
}

void DirichletAnalytic::pup(PUP::er& p) {
  BoundaryCondition::pup(p);
  p | analytic_prescription_;
}
// NOLINTNEXTLINE
PUP::able::PUP_ID DirichletAnalytic::my_PUP_ID = 0;

std::optional<std::string> DirichletAnalytic::dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi,

    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_d_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_ye_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_tau_flux,
    const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> tilde_s_flux,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_phi_flux,

    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        spatial_velocity_one_form,
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        spatial_velocity,
    const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
        inv_spatial_metric,

    const std::optional<
        tnsr::I<DataVector, 3, Frame::Inertial>>& /*face_mesh_velocity*/,
    const tnsr::i<DataVector, 3, Frame::Inertial>& /*normal_covector*/,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords,
    [[maybe_unused]] const double time) const {
  auto boundary_values = call_with_dynamic_type<
      tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataVector>,
                          hydro::Tags::ElectronFraction<DataVector>,
                          hydro::Tags::SpecificInternalEnergy<DataVector>,
                          hydro::Tags::Pressure<DataVector>,
                          hydro::Tags::Temperature<DataVector>,
                          hydro::Tags::SpatialVelocity<DataVector, 3>,
                          hydro::Tags::LorentzFactor<DataVector>,
                          hydro::Tags::MagneticField<DataVector, 3>,
                          hydro::Tags::DivergenceCleaningField<DataVector>,
                          gr::Tags::SpatialMetric<DataVector, 3>,
                          gr::Tags::InverseSpatialMetric<DataVector, 3>,
                          gr::Tags::SqrtDetSpatialMetric<DataVector>,
                          gr::Tags::Lapse<DataVector>,
                          gr::Tags::Shift<DataVector, 3>>,
      grmhd::ValenciaDivClean::InitialData::initial_data_list>(
      analytic_prescription_.get(),
      [&coords, &time](const auto* const initial_data) {
        if constexpr (is_analytic_solution_v<
                          std::decay_t<decltype(*initial_data)>>) {
          return initial_data->variables(
              coords, time,
              tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                         hydro::Tags::ElectronFraction<DataVector>,
                         hydro::Tags::SpecificInternalEnergy<DataVector>,
                         hydro::Tags::Pressure<DataVector>,
                         hydro::Tags::Temperature<DataVector>,
                         hydro::Tags::SpatialVelocity<DataVector, 3>,
                         hydro::Tags::LorentzFactor<DataVector>,
                         hydro::Tags::MagneticField<DataVector, 3>,
                         hydro::Tags::DivergenceCleaningField<DataVector>,
                         gr::Tags::SpatialMetric<DataVector, 3>,
                         gr::Tags::InverseSpatialMetric<DataVector, 3>,
                         gr::Tags::SqrtDetSpatialMetric<DataVector>,
                         gr::Tags::Lapse<DataVector>,
                         gr::Tags::Shift<DataVector, 3>>{});

        } else {
          (void)time;
          return initial_data->variables(
              coords,
              tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                         hydro::Tags::ElectronFraction<DataVector>,
                         hydro::Tags::SpecificInternalEnergy<DataVector>,
                         hydro::Tags::Pressure<DataVector>,
                         hydro::Tags::Temperature<DataVector>,
                         hydro::Tags::SpatialVelocity<DataVector, 3>,
                         hydro::Tags::LorentzFactor<DataVector>,
                         hydro::Tags::MagneticField<DataVector, 3>,
                         hydro::Tags::DivergenceCleaningField<DataVector>,
                         gr::Tags::SpatialMetric<DataVector, 3>,
                         gr::Tags::InverseSpatialMetric<DataVector, 3>,
                         gr::Tags::SqrtDetSpatialMetric<DataVector>,
                         gr::Tags::Lapse<DataVector>,
                         gr::Tags::Shift<DataVector, 3>>{});
        }
      });
  // Recover values from analytic solution/ analytic data calls
  *lapse = get<gr::Tags::Lapse<DataVector>>(boundary_values);
  *shift = get<gr::Tags::Shift<DataVector, 3>>(boundary_values);
  *inv_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(boundary_values);
  *rest_mass_density =
      get<hydro::Tags::RestMassDensity<DataVector>>(boundary_values);
  *electron_fraction =
      get<hydro::Tags::ElectronFraction<DataVector>>(boundary_values);
  *temperature = get<hydro::Tags::Temperature<DataVector>>(boundary_values);
  *spatial_velocity =
      get<hydro::Tags::SpatialVelocity<DataVector, 3>>(boundary_values);
  tenex::evaluate<ti::i>(
      spatial_velocity_one_form,
      (*spatial_velocity)(ti::J) * (get<gr::Tags::SpatialMetric<DataVector, 3>>(
                                       boundary_values)(ti::i, ti::j)));
  // Recover the conservative variables from the primitives
  ConservativeFromPrimitive::apply(
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

  ComputeFluxes::apply(
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

void DirichletAnalytic::fd_ghost(
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        lorentz_factor_times_spatial_velocity,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        magnetic_field,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,

    const gsl::not_null<std::optional<Variables<db::wrap_tags_in<
        Flux, typename grmhd::ValenciaDivClean::System::flux_variables>>>*>
        cell_centered_ghost_fluxes,

    const Direction<3>& direction,

    // fd_interior_temporary_tags
    const Mesh<3>& subcell_mesh,

    // fd_gridless_tags
    [[maybe_unused]] const double time,
    const std::unordered_map<
        std::string,
        std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const ElementMap<3, Frame::Grid>& logical_to_grid_map,
    const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial_map,
    const fd::Reconstructor& reconstructor) const {
  const size_t ghost_zone_size{reconstructor.ghost_zone_size()};

  const auto ghost_logical_coords =
      evolution::dg::subcell::fd::ghost_zone_logical_coordinates(
          subcell_mesh, ghost_zone_size, direction);

  const auto ghost_inertial_coords = grid_to_inertial_map(
      logical_to_grid_map(ghost_logical_coords), time, functions_of_time);

  // Compute FD ghost data with the analytic data or solution
  auto boundary_values = call_with_dynamic_type<
      tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataVector>,
                          hydro::Tags::ElectronFraction<DataVector>,
                          hydro::Tags::Pressure<DataVector>,
                          hydro::Tags::Temperature<DataVector>,
                          hydro::Tags::SpatialVelocity<DataVector, 3>,
                          hydro::Tags::LorentzFactor<DataVector>,
                          hydro::Tags::MagneticField<DataVector, 3>,
                          hydro::Tags::DivergenceCleaningField<DataVector>,
                          hydro::Tags::SpecificInternalEnergy<DataVector>>,
      grmhd::ValenciaDivClean::InitialData::initial_data_list>(
      analytic_prescription_.get(),
      [&ghost_inertial_coords, &time](const auto* const initial_data) {
        using hydro_tags =
            tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                       hydro::Tags::ElectronFraction<DataVector>,
                       hydro::Tags::Pressure<DataVector>,
                       hydro::Tags::Temperature<DataVector>,
                       hydro::Tags::SpatialVelocity<DataVector, 3>,
                       hydro::Tags::LorentzFactor<DataVector>,
                       hydro::Tags::MagneticField<DataVector, 3>,
                       hydro::Tags::DivergenceCleaningField<DataVector>,
                       hydro::Tags::SpecificInternalEnergy<DataVector>>;
        if constexpr (is_analytic_solution_v<
                          std::decay_t<decltype(*initial_data)>>) {
          return initial_data->variables(ghost_inertial_coords, time,
                                         hydro_tags{});
        } else {
          (void)time;
          return initial_data->variables(ghost_inertial_coords, hydro_tags{});
        }
      });

  *rest_mass_density =
      get<hydro::Tags::RestMassDensity<DataVector>>(boundary_values);
  *electron_fraction =
      get<hydro::Tags::ElectronFraction<DataVector>>(boundary_values);
  *temperature = get<hydro::Tags::Temperature<DataVector>>(boundary_values);

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

  if (cell_centered_ghost_fluxes->has_value()) {
    auto metric_boundary_values = call_with_dynamic_type<
        tuples::TaggedTuple<gr::Tags::Lapse<DataVector>,
                            gr::Tags::Shift<DataVector, 3>,
                            gr::Tags::SpatialMetric<DataVector, 3>>,
        grmhd::ValenciaDivClean::InitialData::initial_data_list>(
        analytic_prescription_.get(),
        [&ghost_inertial_coords, &time](const auto* const initial_data) {
          using gr_tags = tmpl::list<gr::Tags::Lapse<DataVector>,
                                     gr::Tags::Shift<DataVector, 3>,
                                     gr::Tags::SpatialMetric<DataVector, 3>>;
          if constexpr (is_analytic_solution_v<
                            std::decay_t<decltype(*initial_data)>>) {
            return initial_data->variables(ghost_inertial_coords, time,
                                           gr_tags{});
          } else {
            (void)time;
            return initial_data->variables(ghost_inertial_coords, gr_tags{});
          }
        });
    auto [sqrt_det_spatial_metric, inverse_spatial_metric] =
        determinant_and_inverse(get<gr::Tags::SpatialMetric<DataVector, 3>>(
            metric_boundary_values));
    get(sqrt_det_spatial_metric) = sqrt(get(sqrt_det_spatial_metric));

    Variables<typename System::variables_tag::tags_list> conserved_vars{
        get(*rest_mass_density).size()};
    ConservativeFromPrimitive::apply(
        make_not_null(&get<Tags::TildeD>(conserved_vars)),
        make_not_null(&get<Tags::TildeYe>(conserved_vars)),
        make_not_null(&get<Tags::TildeTau>(conserved_vars)),
        make_not_null(&get<Tags::TildeS<>>(conserved_vars)),
        make_not_null(&get<Tags::TildeB<>>(conserved_vars)),
        make_not_null(&get<Tags::TildePhi>(conserved_vars)),

        get<hydro::Tags::RestMassDensity<DataVector>>(boundary_values),
        get<hydro::Tags::ElectronFraction<DataVector>>(boundary_values),
        get<hydro::Tags::SpecificInternalEnergy<DataVector>>(boundary_values),
        get<hydro::Tags::Pressure<DataVector>>(boundary_values),
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(boundary_values),
        get<hydro::Tags::LorentzFactor<DataVector>>(boundary_values),
        get<hydro::Tags::MagneticField<DataVector, 3>>(boundary_values),
        sqrt_det_spatial_metric,
        get<gr::Tags::SpatialMetric<DataVector, 3>>(metric_boundary_values),
        get<hydro::Tags::DivergenceCleaningField<DataVector>>(boundary_values));

    ComputeFluxes::apply(
        make_not_null(
            &get<Flux<Tags::TildeD>>(cell_centered_ghost_fluxes->value())),
        make_not_null(
            &get<Flux<Tags::TildeYe>>(cell_centered_ghost_fluxes->value())),
        make_not_null(
            &get<Flux<Tags::TildeTau>>(cell_centered_ghost_fluxes->value())),
        make_not_null(
            &get<Flux<Tags::TildeS<>>>(cell_centered_ghost_fluxes->value())),
        make_not_null(
            &get<Flux<Tags::TildeB<>>>(cell_centered_ghost_fluxes->value())),
        make_not_null(
            &get<Flux<Tags::TildePhi>>(cell_centered_ghost_fluxes->value())),

        get<Tags::TildeD>(conserved_vars), get<Tags::TildeYe>(conserved_vars),
        get<Tags::TildeTau>(conserved_vars),
        get<Tags::TildeS<>>(conserved_vars),
        get<Tags::TildeB<>>(conserved_vars),
        get<Tags::TildePhi>(conserved_vars),

        get<gr::Tags::Lapse<DataVector>>(metric_boundary_values),
        get<gr::Tags::Shift<DataVector, 3>>(metric_boundary_values),
        sqrt_det_spatial_metric,
        get<gr::Tags::SpatialMetric<DataVector, 3>>(metric_boundary_values),
        inverse_spatial_metric,
        get<hydro::Tags::Pressure<DataVector>>(boundary_values),
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(boundary_values),
        get<hydro::Tags::LorentzFactor<DataVector>>(boundary_values),
        get<hydro::Tags::MagneticField<DataVector, 3>>(boundary_values));
  }
}

}  // namespace grmhd::ValenciaDivClean::BoundaryConditions
