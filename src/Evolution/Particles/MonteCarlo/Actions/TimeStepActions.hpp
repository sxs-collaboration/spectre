// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <random>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Particles/MonteCarlo/InverseJacobianInertialToFluidCompute.hpp"
#include "Evolution/Particles/MonteCarlo/MortarData.hpp"
#include "Evolution/Particles/MonteCarlo/NeutrinoInteractionTable.hpp"
#include "Evolution/Particles/MonteCarlo/Tags.hpp"
#include "Evolution/Particles/MonteCarlo/TemplatedLocalFunctions.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"

namespace Particles::MonteCarlo {

/// Mutator advancing neutrinos by a single step
template <size_t EnergyBins, size_t NeutrinoSpecies>
struct TimeStepMutator {
  static const size_t Dim = 3;

  using return_tags =
      tmpl::list<Particles::MonteCarlo::Tags::PacketsOnElement,
                 Particles::MonteCarlo::Tags::RandomNumberGenerator,
                 Particles::MonteCarlo::Tags::DesiredPacketEnergyAtEmission<3>>;
  // To do : check carefully DG vs Subcell quantities... everything should
  // be on the Subcell grid!
  using argument_tags = tmpl::list<
      ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
      hydro::Tags::GrmhdEquationOfState,
      Particles::MonteCarlo::Tags::InteractionRatesTable<EnergyBins,
                                                         NeutrinoSpecies>,
      hydro::Tags::ElectronFraction<DataVector>,
      hydro::Tags::RestMassDensity<DataVector>,
      hydro::Tags::Temperature<DataVector>,
      hydro::Tags::LorentzFactor<DataVector>,
      hydro::Tags::LowerSpatialFourVelocity<DataVector, Dim, Frame::Inertial>,
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<DataVector, Dim, Frame::Inertial>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<Dim>,
                    Frame::Inertial>,
      ::Tags::deriv<gr::Tags::Shift<DataVector, Dim>, tmpl::size_t<Dim>,
                    Frame::Inertial>,
      ::Tags::deriv<gr::Tags::InverseSpatialMetric<DataVector, Dim>,
                    tmpl::size_t<Dim>, Frame::Inertial>,
      gr::Tags::SpatialMetric<DataVector, Dim, Frame::Inertial>,
      gr::Tags::InverseSpatialMetric<DataVector, Dim, Frame::Inertial>,
      gr::Tags::DetSpatialMetric<DataVector>,
      Particles::MonteCarlo::Tags::CellLightCrossingTime<DataVector>,
      evolution::dg::subcell::Tags::Mesh<Dim>,
      evolution::dg::subcell::Tags::Coordinates<Dim, Frame::ElementLogical>,
      domain::Tags::MeshVelocity<Dim>,
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToInertial<Dim>,
      evolution::dg::subcell::fd::Tags::DetInverseJacobianLogicalToInertial,
      domain::Tags::InverseJacobian<Dim + 1, Frame::Inertial, Frame::Fluid>,
      domain::Tags::Jacobian<Dim + 1, Frame::Inertial, Frame::Fluid>,
      Particles::MonteCarlo::Tags::MortarDataTag<Dim>>;

  static void apply(
      const gsl::not_null<std::vector<Packet>*> packets,
      const gsl::not_null<std::mt19937*> random_number_generator,
      const gsl::not_null<std::array<DataVector, NeutrinoSpecies>*>
          single_packet_energy,
      const TimeStepId& current_step_id, const TimeStepId& next_step_id,

      const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
      const NeutrinoInteractionTable<EnergyBins, NeutrinoSpecies>&
          interaction_table,
      const Scalar<DataVector>& electron_fraction,
      const Scalar<DataVector>& rest_mass_density,
      const Scalar<DataVector>& temperature,
      const Scalar<DataVector>& lorentz_factor,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          lower_spatial_four_velocity,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& d_lapse,
      const tnsr::iJ<DataVector, Dim, Frame::Inertial>& d_shift,
      const tnsr::iJJ<DataVector, Dim, Frame::Inertial>& d_inv_spatial_metric,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>& spatial_metric,
      const tnsr::II<DataVector, Dim, Frame::Inertial>& inv_spatial_metric,
      const Scalar<DataVector>& determinant_spatial_metric,
      const Scalar<DataVector>& cell_light_crossing_time, const Mesh<Dim>& mesh,
      const tnsr::I<DataVector, Dim, Frame::ElementLogical>& mesh_coordinates,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>&
          inverse_jacobian_logical_to_inertial,
      const Scalar<DataVector>& det_inverse_jacobian_logical_to_inertial,
      const InverseJacobian<DataVector, Dim + 1, Frame::Inertial, Frame::Fluid>&
          inertial_to_fluid_inverse_jacobian,
      const Jacobian<DataVector, Dim + 1, Frame::Inertial, Frame::Fluid>&
          inertial_to_fluid_jacobian,
      const MortarData<Dim>& mortar_data) {
    // Number of ghost zones for MC is assumed to be 1 for now.
    const size_t num_ghost_zones = 1;
    // Get information stored in various databox containers in
    // the format expected by take_time_step_on_element
    const double start_time = current_step_id.step_time().value();
    const double end_time = next_step_id.step_time().value();
    Scalar<DataVector> det_jacobian_logical_to_inertial(lapse);
    get(det_jacobian_logical_to_inertial) =
        1.0 / get(det_inverse_jacobian_logical_to_inertial);
    const DirectionalIdMap<Dim, std::optional<DataVector>>&
        electron_fraction_ghost = mortar_data.electron_fraction;
    const DirectionalIdMap<Dim, std::optional<DataVector>>&
        baryon_density_ghost = mortar_data.rest_mass_density;
    const DirectionalIdMap<Dim, std::optional<DataVector>>& temperature_ghost =
        mortar_data.temperature;
    const DirectionalIdMap<Dim, std::optional<DataVector>>&
        cell_light_crossing_time_ghost = mortar_data.cell_light_crossing_time;

    TemplatedLocalFunctions<EnergyBins, NeutrinoSpecies> templated_functions;
    templated_functions.take_time_step_on_element(
        packets, random_number_generator, single_packet_energy, start_time,
        end_time, equation_of_state, interaction_table, electron_fraction,
        rest_mass_density, temperature, lorentz_factor,
        lower_spatial_four_velocity, lapse, shift, d_lapse, d_shift,
        d_inv_spatial_metric, spatial_metric, inv_spatial_metric,
        determinant_spatial_metric, cell_light_crossing_time, mesh,
        mesh_coordinates, num_ghost_zones, mesh_velocity,
        inverse_jacobian_logical_to_inertial, det_jacobian_logical_to_inertial,
        inertial_to_fluid_jacobian, inertial_to_fluid_inverse_jacobian,
        electron_fraction_ghost, baryon_density_ghost, temperature_ghost,
        cell_light_crossing_time_ghost);
  }
};

namespace Actions {

/// Action taking a single time step of the Monte-Carlo evolution
/// algorithm, assuming that the fluid and metric data in the ghost
/// zones have been communicated and that packets are on the elements
/// that owns them.
template <size_t EnergyBins, size_t NeutrinoSpecies>
struct TakeTimeStep {
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    ASSERT(db::get<evolution::dg::subcell::Tags::ActiveGrid>(box) ==
               evolution::dg::subcell::ActiveGrid::Subcell,
           "MC assumes that we are using the Subcell grid!");

    db::mutate_apply(TimeStepMutator<EnergyBins, NeutrinoSpecies>{},
                     make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace Actions
}  // namespace Particles::MonteCarlo
