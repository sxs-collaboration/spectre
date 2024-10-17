// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/DirichletAnalytic.hpp"

#include <cstddef>
#include <memory>
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
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/AllSolutions.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Evolution/TypeTraits.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean::BoundaryConditions {
// LCOV_EXCL_START
DirichletAnalytic::DirichletAnalytic(CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}
// LCOV_EXCL_STOP
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

    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_d_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_ye_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_tau_flux,
    const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> tilde_s_flux,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_phi_flux,

    const gsl::not_null<Scalar<DataVector>*> gamma1,
    const gsl::not_null<Scalar<DataVector>*> gamma2,
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
    const Scalar<DataVector>& interior_gamma1,
    const Scalar<DataVector>& interior_gamma2,
    [[maybe_unused]] const double time) const {
  *gamma1 = interior_gamma1;
  *gamma2 = interior_gamma2;

  auto boundary_values = call_with_dynamic_type<
      tuples::TaggedTuple<
          hydro::Tags::RestMassDensity<DataVector>,
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
          gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
          gr::Tags::SpacetimeMetric<DataVector, 3>,
          ::gh::Tags::Pi<DataVector, 3>, ::gh::Tags::Phi<DataVector, 3>>,
      ghmhd::GhValenciaDivClean::InitialData::analytic_solutions_and_data_list>(
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
                         gr::Tags::Shift<DataVector, 3>,
                         gr::Tags::SpacetimeMetric<DataVector, 3>,
                         ::gh::Tags::Pi<DataVector, 3>,
                         ::gh::Tags::Phi<DataVector, 3>>{});

        } else if constexpr (evolution::is_numeric_initial_data_v<
                                 std::decay_t<decltype(*initial_data)>>) {
          ERROR(
              "Cannot currently use numeric initial data as an analytic "
              "prescription in boundary conditions.");
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
                         gr::Tags::Shift<DataVector, 3>,
                         gr::Tags::SpacetimeMetric<DataVector, 3>,
                         ::gh::Tags::Pi<DataVector, 3>,
                         ::gh::Tags::Phi<DataVector, 3>>{});
        }
      });

  *spacetime_metric =
      get<gr::Tags::SpacetimeMetric<DataVector, 3>>(boundary_values);
  *pi = get<::gh::Tags::Pi<DataVector, 3>>(boundary_values);
  *phi = get<::gh::Tags::Phi<DataVector, 3>>(boundary_values);
  *lapse = get<gr::Tags::Lapse<DataVector>>(boundary_values);
  *shift = get<gr::Tags::Shift<DataVector, 3>>(boundary_values);
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

void DirichletAnalytic::fd_ghost(
    const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*>
        spacetime_metric,
    const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> pi,
    const gsl::not_null<tnsr::iaa<DataVector, 3, Frame::Inertial>*> phi,
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        lorentz_factor_times_spatial_velocity,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        magnetic_field,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
    const Direction<3>& direction,

    // fd_interior_temporary_tags
    const Mesh<3>& subcell_mesh,

    // fd_gridless_tags
    const double time,
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
                          hydro::Tags::SpecificInternalEnergy<DataVector>,
                          hydro::Tags::Temperature<DataVector>,
                          hydro::Tags::SpatialVelocity<DataVector, 3>,
                          hydro::Tags::LorentzFactor<DataVector>,
                          hydro::Tags::MagneticField<DataVector, 3>,
                          hydro::Tags::DivergenceCleaningField<DataVector>,
                          gr::Tags::SpacetimeMetric<DataVector, 3>,
                          ::gh::Tags::Pi<DataVector, 3>,
                          ::gh::Tags::Phi<DataVector, 3>>,
      ghmhd::GhValenciaDivClean::InitialData::analytic_solutions_and_data_list>(
      analytic_prescription_.get(),
      [&ghost_inertial_coords, &time](const auto* const initial_data) {
        using hydro_and_spacetime_tags =
            tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                       hydro::Tags::ElectronFraction<DataVector>,
                       hydro::Tags::SpecificInternalEnergy<DataVector>,
                       hydro::Tags::Temperature<DataVector>,
                       hydro::Tags::SpatialVelocity<DataVector, 3>,
                       hydro::Tags::LorentzFactor<DataVector>,
                       hydro::Tags::MagneticField<DataVector, 3>,
                       hydro::Tags::DivergenceCleaningField<DataVector>,
                       gr::Tags::SpacetimeMetric<DataVector, 3>,
                       ::gh::Tags::Pi<DataVector, 3>,
                       ::gh::Tags::Phi<DataVector, 3>>;
        if constexpr (is_analytic_solution_v<
                          std::decay_t<decltype(*initial_data)>>) {
          return initial_data->variables(ghost_inertial_coords, time,
                                         hydro_and_spacetime_tags{});
        } else if constexpr (evolution::is_numeric_initial_data_v<
                                 std::decay_t<decltype(*initial_data)>>) {
          ERROR(
              "Cannot currently use numeric initial data as an analytic "
              "prescription for boundary conditions.");
        } else {
          (void)time;
          return initial_data->variables(ghost_inertial_coords,
                                         hydro_and_spacetime_tags{});
        }
      });
  *spacetime_metric =
      get<gr::Tags::SpacetimeMetric<DataVector, 3>>(boundary_values);
  *pi = get<::gh::Tags::Pi<DataVector, 3>>(boundary_values);
  *phi = get<::gh::Tags::Phi<DataVector, 3>>(boundary_values);
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
}

}  // namespace grmhd::GhValenciaDivClean::BoundaryConditions
