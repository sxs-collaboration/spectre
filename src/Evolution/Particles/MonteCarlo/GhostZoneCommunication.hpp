// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/NeighborMesh.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Interpolators.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Particles/MonteCarlo/GhostZoneCommunicationStep.hpp"
#include "Evolution/Particles/MonteCarlo/GhostZoneCommunicationTags.hpp"
#include "Evolution/Particles/MonteCarlo/MortarData.hpp"
#include "Evolution/Particles/MonteCarlo/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace Tags {
struct TimeStepId;
}  // namespace Tags
/// \endcond

namespace Particles::MonteCarlo::Actions {

/// Mutator to get required volume data for communication
/// before a MC step; i.e. data sent from live points
/// to ghost points in neighbors.
struct GhostDataMutatorPreStep {
  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<
      hydro::Tags::RestMassDensity<DataVector>,
      hydro::Tags::ElectronFraction<DataVector>,
      hydro::Tags::Temperature<DataVector>,
      Particles::MonteCarlo::Tags::CellLightCrossingTime<DataVector>>;
  static const size_t number_of_vars = 4;

  static DataVector apply(const Scalar<DataVector>& rest_mass_density,
                          const Scalar<DataVector>& electron_fraction,
                          const Scalar<DataVector>& temperature,
                          const Scalar<DataVector>& cell_light_crossing_time) {
    const size_t dv_size = get(rest_mass_density).size();
    DataVector buffer{dv_size * number_of_vars};
    std::copy(
        get(rest_mass_density).data(),
        std::next(get(rest_mass_density).data(), static_cast<int>(dv_size)),
        buffer.data());
    std::copy(
        get(electron_fraction).data(),
        std::next(get(electron_fraction).data(), static_cast<int>(dv_size)),
        std::next(buffer.data(), static_cast<int>(dv_size)));
    std::copy(get(temperature).data(),
              std::next(get(temperature).data(), static_cast<int>(dv_size)),
              std::next(buffer.data(), static_cast<int>(dv_size * 2)));
    std::copy(get(cell_light_crossing_time).data(),
              std::next(get(cell_light_crossing_time).data(),
                        static_cast<int>(dv_size)),
              std::next(buffer.data(), static_cast<int>(dv_size * 3)));
    return buffer;
  }
};

/// Mutator to get required volume data for communication
/// after a MC step; i.e. data sent from ghost points
/// in neighbors to live points evolving the fluid. The
/// data contains information about the back reaction of
/// neutrinos on the fluid
template <size_t Dim>
struct GhostDataMutatorPostStep {
  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<Particles::MonteCarlo::Tags::CouplingTildeTau<DataVector>,
                 Particles::MonteCarlo::Tags::CouplingTildeRhoYe<DataVector>,
                 Particles::MonteCarlo::Tags::CouplingTildeS<DataVector, Dim>>;
  static const size_t number_of_components = 2 + Dim;

  static DataVector apply(const Scalar<DataVector>& coupling_tilde_tau,
                          const Scalar<DataVector>& coupling_tilde_rho_ye,
                          const tnsr::i<DataVector, Dim>& coupling_tilde_s) {
    const size_t dv_size = get(coupling_tilde_tau).size();
    DataVector buffer{dv_size * number_of_components};
    std::copy(
        get(coupling_tilde_tau).data(),
        std::next(get(coupling_tilde_tau).data(), static_cast<int>(dv_size)),
        buffer.data());
    std::copy(
        get(coupling_tilde_rho_ye).data(),
        std::next(get(coupling_tilde_rho_ye).data(), static_cast<int>(dv_size)),
        std::next(buffer.data(), static_cast<int>(dv_size)));
    for (size_t d = 0; d < Dim; d++) {
      std::copy(
          coupling_tilde_s.get(d).data(),
          std::next(coupling_tilde_s.get(d).data(), static_cast<int>(dv_size)),
          std::next(buffer.data(), static_cast<int>(dv_size * (2 + d))));
    }
    return buffer;
  }
};

/// Mutator that returns packets currently in ghost zones in a
/// DirectionMap<Dim,std::vector<Particles::MonteCarlo::Packet>>
/// and remove them from the list of packets of the current
/// element
template <size_t Dim>
struct GhostDataMcPackets {
  using return_tags = tmpl::list<Particles::MonteCarlo::Tags::PacketsOnElement>;
  using argument_tags = tmpl::list<::domain::Tags::Element<Dim>>;

  static DirectionMap<Dim, std::vector<Particles::MonteCarlo::Packet>> apply(
      const gsl::not_null<std::vector<Particles::MonteCarlo::Packet>*> packets,
      const Element<Dim>& element) {
    DirectionMap<Dim, std::vector<Particles::MonteCarlo::Packet>> output{};
    for (const auto& [direction, neighbors_in_direction] :
         element.neighbors()) {
      output[direction] = std::vector<Particles::MonteCarlo::Packet>{};
    }
    // Loop over packets. If a packet is out of the current element,
    // remove it from the element and copy it to the appropriate
    // neighbor list if it exists.
    // Note: At this point, we only handle simple boundaries
    // (one neighbor per direction, with the logical coordinates
    // being transformed by adding/removing 2). We assume that
    // packets that go out of the element but do not match any
    // neighbor have reached the domain boundary and can be removed.
    size_t n_packets = packets->size();
    for (size_t p = 0; p < n_packets; p++) {
      Packet& packet = (*packets)[p];
      std::optional<Direction<Dim>> max_distance_direction = std::nullopt;
      double max_distance = 0.0;
      // TO DO: Deal with dimensions higher than the grid dimension
      // e.g. for axisymmetric evolution
      for (size_t d = 0; d < Dim; d++) {
        if (packet.coordinates.get(d) < -1.0) {
          const double distance = -1.0 - packet.coordinates.get(d);
          if (max_distance < distance) {
            max_distance = distance;
            max_distance_direction = Direction<Dim>(d, Side::Lower);
          }
        }
        if (packet.coordinates.get(d) > 1.0) {
          const double distance = packet.coordinates.get(d) - 1.0;
          if (max_distance < distance) {
            max_distance = distance;
            max_distance_direction = Direction<Dim>(d, Side::Upper);
          }
        }
      }
      // If a packet should be moved along an edge / corner, we move
      // it into the direct neighbor it is effectively closest to
      // (in topological coordinates). This may need improvements.
      if (max_distance_direction != std::nullopt) {
        const size_t d = max_distance_direction.value().dimension();
        const Side side = max_distance_direction.value().side();
        packet.coordinates.get(d) += (side == Side::Lower) ? (2.0) : (-2.0);
        // Corner/edge treatment; move packet in a live point for now
        // Definitely needs improvement...
        for (size_t dd = 0; dd < Dim; dd++) {
          if (dd != d && packet.coordinates.get(dd) < -1.0) {
            packet.coordinates.get(dd) = -2.0 - packet.coordinates.get(dd);
          }
          if (dd != d && packet.coordinates.get(dd) > 1.0) {
            packet.coordinates.get(dd) = 2.0 - packet.coordinates.get(dd);
          }
        }
        if (output.contains(max_distance_direction.value())) {
          output[max_distance_direction.value()].push_back(packet);
        }
        std::swap((*packets)[p], (*packets)[n_packets - 1]);
        packets->pop_back();
        p--;
        n_packets--;
      }
    }
    return output;
  }
};

/// Action responsible for the Send operation of ghost
/// zone communication for Monte-Carlo transport.
/// If CommStep == PreStep, this sends the fluid and
/// metric variables needed for evolution of MC packets.
/// If CommStep == PostStep, this sends packets that
/// have moved from one element to another as well
/// as coupling information from the ghost zone to the
/// live points.
template <size_t Dim, bool LocalTimeStepping,
          Particles::MonteCarlo::CommunicationStep CommStep>
struct SendDataForMcCommunication {
  using inbox_tags =
      tmpl::list<Particles::MonteCarlo::McGhostZoneDataInboxTag<Dim, CommStep>>;

  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(not LocalTimeStepping,
                  "Monte Carlo is not tested with local time stepping.");

    ASSERT(db::get<evolution::dg::subcell::Tags::ActiveGrid>(box) ==
               evolution::dg::subcell::ActiveGrid::Subcell,
           "The SendDataForReconstructionPreStep action in MC "
           "can only be called when Subcell is the active scheme.");
    db::mutate<Particles::MonteCarlo::Tags::McGhostZoneDataTag<Dim>>(
        [](const auto ghost_data_ptr) {
          // Clear the previous neighbor data and add current local data
          ghost_data_ptr->clear();
        },
        make_not_null(&box));

    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    const size_t ghost_zone_size = 1;

    const Element<Dim>& element = db::get<::domain::Tags::Element<Dim>>(box);
    const Mesh<Dim>& subcell_mesh =
        db::get<evolution::dg::subcell::Tags::Mesh<Dim>>(box);
    const TimeStepId& time_step_id = db::get<::Tags::TimeStepId>(box);
    Index<Dim> extents_with_ghost_zone = subcell_mesh.extents();
    for (size_t d = 0; d < Dim; d++) {
      extents_with_ghost_zone[d] += 2 * ghost_zone_size;
    }

    // Fill volume data that should be fed to neighbor ghost zones
    // PreStep, we send fluid data from the live points to the ghost points.
    // PostStep, we send coupling data from the ghost points to the live
    // points (note the different DataVector sizes; the PostStep data has
    // different dimensions because it is taken from DataVectors including
    // ghost points.
    DataVector volume_data_to_slice =
        CommStep == Particles::MonteCarlo::CommunicationStep::PreStep
            ? db::mutate_apply(GhostDataMutatorPreStep{}, make_not_null(&box))
            : db::mutate_apply(GhostDataMutatorPostStep<Dim>{},
                               make_not_null(&box));
    const DirectionMap<Dim, DataVector> all_sliced_data =
        CommStep == Particles::MonteCarlo::CommunicationStep::PreStep
            ? evolution::dg::subcell::slice_data(
                  volume_data_to_slice, subcell_mesh.extents(), ghost_zone_size,
                  element.internal_boundaries(), 0,
                  db::get<evolution::dg::subcell::Tags::
                              InterpolatorsFromFdToNeighborFd<Dim>>(box))
            : evolution::dg::subcell::slice_data(
                  volume_data_to_slice, extents_with_ghost_zone,
                  ghost_zone_size, element.internal_boundaries(), 0,
                  db::get<evolution::dg::subcell::Tags::
                              InterpolatorsFromFdToNeighborFd<Dim>>(box));
    const DirectionMap<Dim, std::vector<Particles::MonteCarlo::Packet>>
        all_packets_ghost_zone =
            CommStep == Particles::MonteCarlo::CommunicationStep::PostStep
                ? db::mutate_apply(GhostDataMcPackets<Dim>{},
                                   make_not_null(&box))
                : DirectionMap<Dim,
                               std::vector<Particles::MonteCarlo::Packet>>{};

    for (const auto& [direction, neighbors_in_direction] :
         element.neighbors()) {
      const auto& orientation = neighbors_in_direction.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());
      for (const ElementId<Dim>& neighbor : neighbors_in_direction) {
        std::optional<std::vector<Particles::MonteCarlo::Packet>>
            packets_to_send = std::nullopt;
        DataVector subcell_data_to_send{};

        // PreStep and PostStep both send slice data to neighboring domains
        const auto& sliced_data_in_direction = all_sliced_data.at(direction);
        subcell_data_to_send = DataVector{sliced_data_in_direction.size()};
        std::copy(sliced_data_in_direction.begin(),
                  sliced_data_in_direction.end(), subcell_data_to_send.begin());
        if (CommStep == Particles::MonteCarlo::CommunicationStep::PostStep) {
          packets_to_send = all_packets_ghost_zone.at(direction);
          if (packets_to_send.value().empty()) {
            packets_to_send = std::nullopt;
          }
        }

        McGhostZoneData<Dim> data{subcell_data_to_send, packets_to_send};

        Parallel::receive_data<
            Particles::MonteCarlo::McGhostZoneDataInboxTag<Dim, CommStep>>(
            receiver_proxy[neighbor], time_step_id,
            std::pair{DirectionalId<Dim>{direction_from_neighbor, element.id()},
                      std::move(data)});
      }
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/// Action responsible for the Receive operation of ghost
/// zone communication for Monte-Carlo transport.
/// If CommStep == PreStep, this gets the fluid and
/// metric variables needed for evolution of MC packets.
/// If CommStep == PostStep, this gets packets that
/// have moved into the current element.
/// The current code does not yet deal with data required
/// to calculate the backreaction on the fluid.
template <size_t Dim, CommunicationStep CommStep>
struct ReceiveDataForMcCommunication {
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const Element<Dim>& element = db::get<::domain::Tags::Element<Dim>>(box);
    const auto number_of_expected_messages = element.neighbors().size();
    if (UNLIKELY(number_of_expected_messages == 0)) {
      // We have no neighbors, so just continue without doing any work
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    const auto& current_time_step_id = db::get<::Tags::TimeStepId>(box);
    std::map<TimeStepId, DirectionalIdMap<
                             Dim, Particles::MonteCarlo::McGhostZoneData<Dim>>>&
        inbox = tuples::get<Particles::MonteCarlo::McGhostZoneDataInboxTag<
            Metavariables::volume_dim, CommStep>>(inboxes);
    const auto& received = inbox.find(current_time_step_id);
    // Check we have at least some data from correct time, and then check that
    // we have received all data
    if (received == inbox.end() or
        received->second.size() != number_of_expected_messages) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }

    // Now that we have received all the data, copy it over as needed.
    DirectionalIdMap<Dim, Particles::MonteCarlo::McGhostZoneData<Dim>>
        received_data = std::move(inbox[current_time_step_id]);
    inbox.erase(current_time_step_id);

    // Copy the received PreStep data in the MortarData container
    if (CommStep == Particles::MonteCarlo::CommunicationStep::PreStep) {
      db::mutate<Particles::MonteCarlo::Tags::MortarDataTag<Dim>>(
          [&element,
           &received_data](const gsl::not_null<MortarData<Dim>*> mortar_data) {
            for (const auto& [direction, neighbors_in_direction] :
                 element.neighbors()) {
              for (const auto& neighbor : neighbors_in_direction) {
                DirectionalId<Dim> directional_element_id{direction, neighbor};
                const DataVector& received_data_direction =
                    received_data[directional_element_id]
                        .ghost_zone_hydro_variables;
                if (mortar_data->rest_mass_density[directional_element_id] ==
                    std::nullopt) {
                  continue;
                }
                const size_t mortar_data_size =
                    mortar_data->rest_mass_density[directional_element_id]
                        .value()
                        .size();
                ASSERT(received_data_direction.size() == mortar_data_size * 4,
                       "Inconsistent sizes between inbox and mortar data");
                std::copy(received_data_direction.data(),
                          std::next(received_data_direction.data(),
                                    static_cast<int>(mortar_data_size)),
                          mortar_data->rest_mass_density[directional_element_id]
                              .value()
                              .data());
                std::copy(std::next(received_data_direction.data(),
                                    static_cast<int>(mortar_data_size)),
                          std::next(received_data_direction.data(),
                                    static_cast<int>(mortar_data_size * 2)),
                          mortar_data->electron_fraction[directional_element_id]
                              .value()
                              .data());
                std::copy(std::next(received_data_direction.data(),
                                    static_cast<int>(mortar_data_size * 2)),
                          std::next(received_data_direction.data(),
                                    static_cast<int>(mortar_data_size * 3)),
                          mortar_data->temperature[directional_element_id]
                              .value()
                              .data());
                std::copy(std::next(received_data_direction.data(),
                                    static_cast<int>(mortar_data_size * 3)),
                          std::next(received_data_direction.data(),
                                    static_cast<int>(mortar_data_size * 4)),
                          mortar_data
                              ->cell_light_crossing_time[directional_element_id]
                              .value()
                              .data());
              }
            }
          },
          make_not_null(&box));
    } else {
      const Mesh<Dim>& subcell_mesh =
          db::get<evolution::dg::subcell::Tags::Mesh<Dim>>(box);
      db::mutate<Particles::MonteCarlo::Tags::GhostZoneCouplingDataTag<Dim>,
                 Particles::MonteCarlo::Tags::PacketsOnElement>(
          [&element, &received_data, &subcell_mesh](
              const gsl::not_null<GhostZoneCouplingData<Dim>*> coupling_data,
              const gsl::not_null<std::vector<Particles::MonteCarlo::Packet>*>
                  packet_list) {
            // Two parts here: first deal with coupling data brought back
            // from ghost zones to live points; then handle packet
            // communication.
            const Index<Dim>& extents = subcell_mesh.extents();
            for (const auto& [direction, neighbors_in_direction] :
                 element.neighbors()) {
              for (const auto& neighbor : neighbors_in_direction) {
                DirectionalId<Dim> directional_element_id{direction, neighbor};
                // Move boundary data to the coupling_data structure for later
                // use
                const DataVector& received_data_direction =
                    received_data[directional_element_id]
                        .ghost_zone_hydro_variables;
                if (coupling_data->coupling_tilde_tau[directional_element_id] !=
                    std::nullopt) {
                  const size_t coupling_data_size =
                      coupling_data->coupling_tilde_tau[directional_element_id]
                          .value()
                          .size();
                  ASSERT(received_data_direction.size() ==
                             coupling_data_size * (2 + Dim),
                         "Inconsistent sizes between inbox and coupling data");
                  std::copy(
                      received_data_direction.data(),
                      std::next(received_data_direction.data(),
                                static_cast<int>(coupling_data_size)),
                      coupling_data->coupling_tilde_tau[directional_element_id]
                          .value()
                          .data());
                  std::copy(std::next(received_data_direction.data(),
                                      static_cast<int>(coupling_data_size)),
                            std::next(received_data_direction.data(),
                                      static_cast<int>(coupling_data_size * 2)),
                            coupling_data
                                ->coupling_tilde_rho_ye[directional_element_id]
                                .value()
                                .data());
                  for (size_t d = 0; d < Dim; d++) {
                    std::copy(
                        std::next(
                            received_data_direction.data(),
                            static_cast<int>(coupling_data_size * (2 + d))),
                        std::next(
                            received_data_direction.data(),
                            static_cast<int>(coupling_data_size * (3 + d))),
                        coupling_data->coupling_tilde_s[directional_element_id]
                            .value()
                            .get(d)
                            .data());
                  }
                }
                // Get packet data
                std::optional<std::vector<Particles::MonteCarlo::Packet>>&
                    received_data_packets =
                        received_data[directional_element_id]
                            .packets_entering_this_element;
                if (received_data_packets == std::nullopt) {
                  continue;
                } else {
                  const size_t n_packets = received_data_packets.value().size();
                  for (size_t p = 0; p < n_packets; p++) {
                    // Find closest grid point to received packet at current
                    // time, using extents for live points only.
                    Packet& packet = received_data_packets.value()[p];
                    Index<Dim> closest_point_index_3d;
                    for (size_t d = 0; d < Dim; d++) {
                      closest_point_index_3d[d] =
                          std::floor((packet.coordinates[d] + 1.0) *
                                     static_cast<double>(extents[d]) / 2.0);
                      if (UNLIKELY(closest_point_index_3d[d] == extents[d])) {
                        closest_point_index_3d[d]--;
                      }
                    }
                    packet.index_of_closest_grid_point =
                        collapsed_index(closest_point_index_3d, extents);
                    // Now add packets to list in current element
                    packet_list->push_back(received_data_packets.value()[p]);
                  }
                }
              }
            }
          },
          make_not_null(&box));
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace Particles::MonteCarlo::Actions
