// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <random>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Variables.hpp"
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
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativeSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
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
                 Particles::MonteCarlo::Tags::CouplingTildeTau<DataVector>,
                 Particles::MonteCarlo::Tags::CouplingTildeRhoYe<DataVector>,
                 Particles::MonteCarlo::Tags::CouplingTildeS<DataVector, Dim>,
                 Particles::MonteCarlo::Tags::RandomNumberGenerator,
                 Particles::MonteCarlo::Tags::DesiredPacketEnergyAtEmission<
                     NeutrinoSpecies>>;
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
      hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>,
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<DataVector, Dim, Frame::Inertial>,
      gh::Tags::Phi<DataVector, 3, Frame::Inertial>,
      gr::Tags::SpatialMetric<DataVector, Dim, Frame::Inertial>,
      gr::Tags::InverseSpatialMetric<DataVector, Dim, Frame::Inertial>,
      gr::Tags::SqrtDetSpatialMetric<DataVector>,
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
      const gsl::not_null<Scalar<DataVector>*> coupling_tilde_tau,
      const gsl::not_null<Scalar<DataVector>*> coupling_tilde_rho_ye,
      const gsl::not_null<tnsr::i<DataVector, Dim>*> coupling_tilde_s,
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
      const tnsr::I<DataVector, Dim, Frame::Inertial>& spatial_velocity,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,

      const tnsr::ii<DataVector, Dim, Frame::Inertial>& spatial_metric,
      const tnsr::II<DataVector, Dim, Frame::Inertial>& inv_spatial_metric,
      const Scalar<DataVector>& sqrt_determinant_spatial_metric,
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

    // Calculate temporary tensors needed for MC evolution
    using deriv_lapse = ::Tags::deriv<gr::Tags::Lapse<DataVector>,
                                      tmpl::size_t<3>, Frame::Inertial>;
    using deriv_shift = ::Tags::deriv<gr::Tags::Shift<DataVector, 3>,
                                      tmpl::size_t<3>, Frame::Inertial>;
    using deriv_spatial_metric =
        ::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>, tmpl::size_t<3>,
                      Frame::Inertial>;
    using deriv_inverse_spatial_metric =
        ::Tags::deriv<gr::Tags::InverseSpatialMetric<DataVector, 3>,
                      tmpl::size_t<3>, Frame::Inertial>;
    using temporary_tags = tmpl::list<
        hydro::Tags::LowerSpatialFourVelocity<DataVector, Dim, Frame::Inertial>,
        gr::Tags::SpacetimeNormalVector<DataVector, 3>,
        gr::Tags::InverseSpacetimeMetric<DataVector, 3>, deriv_lapse,
        deriv_shift, deriv_spatial_metric, deriv_inverse_spatial_metric>;
    Variables<temporary_tags> temp_tags{mesh.number_of_grid_points(), 0.0};

    // u_i = \gamma_{ij} v^j W
    auto& lower_spatial_four_velocity =
        get<hydro::Tags::LowerSpatialFourVelocity<DataVector, Dim,
                                                  Frame::Inertial>>(temp_tags);
    raise_or_lower_index(make_not_null(&lower_spatial_four_velocity),
                         spatial_velocity, spatial_metric);
    for (size_t i = 0; i < Dim; i++) {
      lower_spatial_four_velocity.get(i) *= get(lorentz_factor);
    }
    // For the metric, we adapt the calculations performed for the time
    // derivative of in GhGrMhd. First get n^a and g^ab
    auto& spacetime_normal_vector =
        get<gr::Tags::SpacetimeNormalVector<DataVector, 3>>(temp_tags);
    auto& inv_spacetime_metric =
        get<gr::Tags::InverseSpacetimeMetric<DataVector, 3>>(temp_tags);
    gr::spacetime_normal_vector(make_not_null(&spacetime_normal_vector), lapse,
                                shift);
    gr::inverse_spacetime_metric(make_not_null(&inv_spacetime_metric), lapse,
                                 shift, inv_spatial_metric);

    auto& d_lapse = get<deriv_lapse>(temp_tags);
    auto& d_shift = get<deriv_shift>(temp_tags);
    // Temporary store phi_iab n^a n^b in d_lapse. This is phi_two_normals in GH
    for (size_t i = 0; i < Dim; i++) {
      for (size_t a = 0; a < Dim + 1; a++) {
        for (size_t b = 0; b < Dim + 1; b++) {
          d_lapse.get(i) += phi.get(i, a, b) * spacetime_normal_vector.get(a) *
                            spacetime_normal_vector.get(b);
        }
      }
    }
    // Shift derivative using stored quantity in d_lapse
    // We use d_i shift^j =
    // (g^{j+1 b} phi_{iba} n^a + n^{j+1} phi_{iab} n^a n_b) * lapse
    // as in TimeDerivative.hpp in GhGrMhd
    for (size_t i = 0; i < Dim; i++) {
      for (size_t j = 0; j < Dim; j++) {
        d_shift.get(i, j) +=
            d_lapse.get(i) * spacetime_normal_vector.get(j + 1);
        for (size_t a = 0; a < Dim + 1; a++) {
          for (size_t b = 0; b < Dim + 1; b++) {
            d_shift.get(i, j) += inv_spacetime_metric.get(j + 1, b) *
                                 phi.get(i, b, a) *
                                 spacetime_normal_vector.get(a);
          }
        }
        d_shift.get(i, j) *= get(lapse);
      }
    }
    // Now use d_i lapse = - lapse * 0.5 * phi_{iab} n^a n^b
    // As we already stored phi_{iab} n^a n^b in d_i lapse,
    // we just multiply by (-0.5 * lapse)
    for (size_t i = 0; i < Dim; i++) {
      d_lapse.get(i) *= (-0.5) * get(lapse);
    }

    // Extract d_i \gamma_{jk} from phi_{i,j+1,k+1}
    auto& d_spatial_metric = get<deriv_spatial_metric>(temp_tags);
    for (size_t i = 0; i < Dim; i++) {
      for (size_t j = 0; j < Dim; j++) {
        for (size_t k = j; k < Dim; k++) {
          d_spatial_metric.get(i, j, k) = phi.get(i, j + 1, k + 1);
        }
      }
    }

    auto& d_inv_spatial_metric = get<deriv_inverse_spatial_metric>(temp_tags);
    gr::deriv_inverse_spatial_metric(make_not_null(&d_inv_spatial_metric),
                                     inv_spatial_metric, d_spatial_metric);

    TemplatedLocalFunctions<EnergyBins, NeutrinoSpecies> templated_functions;
    templated_functions.take_time_step_on_element(
        packets, coupling_tilde_tau, coupling_tilde_rho_ye, coupling_tilde_s,
        random_number_generator, single_packet_energy, start_time, end_time,
        equation_of_state, interaction_table, electron_fraction,
        rest_mass_density, temperature, lorentz_factor,
        lower_spatial_four_velocity, lapse, shift, d_lapse, d_shift,
        d_inv_spatial_metric, spatial_metric, inv_spatial_metric,
        sqrt_determinant_spatial_metric, cell_light_crossing_time, mesh,
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
