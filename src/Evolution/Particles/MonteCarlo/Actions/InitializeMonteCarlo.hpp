// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <random>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Evolution/Particles/MonteCarlo/GhostZoneCommunicationTags.hpp"
#include "Evolution/Particles/MonteCarlo/MonteCarloOptions.hpp"
#include "Evolution/Particles/MonteCarlo/MortarData.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "Evolution/Particles/MonteCarlo/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace evolution::initial_data::Tags {
struct InitialData;
}  // namespace evolution::initial_data::Tags

namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples

namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
/// \endcond

namespace Initialization::Actions {

/// \ingroup InitializationGroup
/// \brief Allocate variables needed for evolution of Monte Carlo transport
///
/// Uses:
/// - evolution::dg::subcell::Tags::Mesh<dim>
/// - evolution::dg::subcell::Tags::Coordinates<dim, Frame::Inertial>
/// - evolution::initial_data::Tags::InitialData
/// - Particles::MonteCarlo::Tags::MonteCarloOptions<EnergyBins,
/// NeutrinoSpecies>
/// - domain::Tags::Element<dim>
///
/// DataBox changes:
/// - Adds:
///   * Particles::MonteCarlo::Tags::PacketsOnElement
///   * Particles::MonteCarlo::Tags::RandomNumberGenerator
///   * Particles::MonteCarlo::Tags::DesiredPacketEnergyAtEmission<
///                                  NeutrinoSpecies>
///   * Background hydro variables
///   * Particles::MonteCarlo::Tags::CouplingTildeTau<DataVector>
///   * Particles::MonteCarlo::Tags::CouplingTildeRhoYe<DataVector>
///   * Particles::MonteCarlo::Tags::CouplingTildeS<DataVector,dim>
///   * Particles::MonteCarlo::Tags::MortarDataTag<dim>
///   * Particles::MonteCarlo::Tags::GhostZoneCouplingData<dim>
///   * Particles::MonteCarlo::Tags::McGhostZoneDataTag<dim>
///
/// - Removes: nothing
/// - Modifies: nothing
template <typename System, size_t EnergyBins, size_t NeutrinoSpecies>
struct InitializeMCTags {
 public:
  using hydro_variables_tag = typename System::hydro_variables_tag;

  static constexpr size_t dim = System::volume_dim;
  using simple_tags =
      tmpl::list<Particles::MonteCarlo::Tags::PacketsOnElement,
                 Particles::MonteCarlo::Tags::RandomNumberGenerator,
                 Particles::MonteCarlo::Tags::DesiredPacketEnergyAtEmission<
                     NeutrinoSpecies>,
                 hydro_variables_tag,
                 Particles::MonteCarlo::Tags::CouplingTildeTau<DataVector>,
                 Particles::MonteCarlo::Tags::CouplingTildeRhoYe<DataVector>,
                 Particles::MonteCarlo::Tags::CouplingTildeS<DataVector, dim>,
                 Particles::MonteCarlo::Tags::MortarDataTag<dim>,
                 Particles::MonteCarlo::Tags::GhostZoneCouplingDataTag<dim>,
                 Particles::MonteCarlo::Tags::McGhostZoneDataTag<dim>,
                 evolution::dg::subcell::Tags::ActiveGrid>;

  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if (db::get<evolution::dg::subcell::Tags::ActiveGrid>(box) !=
        evolution::dg::subcell::ActiveGrid::Subcell) {
      ERROR("MC requires all elements to use Subcell");
    }
    const Mesh<dim>& mesh =
        db::get<evolution::dg::subcell::Tags::Mesh<dim>>(box);
    const size_t num_grid_points = mesh.number_of_grid_points();
    // Number of ghost zones for MC is assumed to be 1 for now.
    const size_t num_ghost_zones = 1;
    size_t mesh_size_with_ghost_zones = 1;
    for (size_t d = 0; d < dim; d++) {
      mesh_size_with_ghost_zones *= (mesh.extents()[d] + 2 * num_ghost_zones);
    }
    const DataVector zero_dv_with_ghost_zones(mesh_size_with_ghost_zones, 0.0);
    const Scalar<DataVector> zero_scalar_with_ghost_zones =
        make_with_value<Scalar<DataVector>>(zero_dv_with_ghost_zones, 0.0);
    const tnsr::i<DataVector, dim, Frame::Inertial> zero_tnsr_with_ghost_zones =
        make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
            zero_dv_with_ghost_zones, 0.0);
    using derived_classes =
        tmpl::at<typename Metavariables::factory_creation::factory_classes,
                 evolution::initial_data::InitialData>;
    using HydroVars = typename hydro_variables_tag::type;
    call_with_dynamic_type<void, derived_classes>(
        &db::get<evolution::initial_data::Tags::InitialData>(box),
        [&box, &num_grid_points](const auto* const data_or_solution) {
          static constexpr size_t dim = System::volume_dim;
          const double initial_time = db::get<::Tags::Time>(box);
          const auto& inertial_coords = db::get<
              evolution::dg::subcell::Tags::Coordinates<dim, Frame::Inertial>>(
              box);
          // Get hydro variables
          HydroVars hydro_variables{num_grid_points};
          hydro_variables.assign_subset(evolution::Initialization::initial_data(
              *data_or_solution, inertial_coords, initial_time,
              typename hydro_variables_tag::tags_list{}));
          Initialization::mutate_assign<tmpl::list<hydro_variables_tag>>(
              make_not_null(&box), std::move(hydro_variables));
        });

    Initialization::mutate_assign<
        tmpl::list<Particles::MonteCarlo::Tags::CouplingTildeTau<DataVector>>>(
        make_not_null(&box), zero_scalar_with_ghost_zones);
    Initialization::mutate_assign<tmpl::list<
        Particles::MonteCarlo::Tags::CouplingTildeRhoYe<DataVector>>>(
        make_not_null(&box), zero_scalar_with_ghost_zones);
    Initialization::mutate_assign<tmpl::list<
        Particles::MonteCarlo::Tags::CouplingTildeS<DataVector, dim>>>(
        make_not_null(&box), zero_tnsr_with_ghost_zones);

    // Read global options for Monte-Carlo evolution
    const auto mc_options = db::get<
        Particles::MonteCarlo::Tags::MonteCarloOptions<NeutrinoSpecies>>(box);
    const auto& initial_packet_energy = mc_options.get_initial_packet_energy();

    typename Particles::MonteCarlo::Tags::PacketsOnElement::type all_packets;
    Initialization::mutate_assign<
        tmpl::list<Particles::MonteCarlo::Tags::PacketsOnElement>>(
        make_not_null(&box), std::move(all_packets));

    const unsigned long seed = std::random_device{}();
    typename Particles::MonteCarlo::Tags::RandomNumberGenerator::type rng(seed);

    Initialization::mutate_assign<
        tmpl::list<Particles::MonteCarlo::Tags::RandomNumberGenerator>>(
        make_not_null(&box), std::move(rng));

    // Initial energy of packets, read from MC options
    typename Particles::MonteCarlo::Tags::DesiredPacketEnergyAtEmission<
        NeutrinoSpecies>::type packet_energy_at_emission =
        make_with_value<std::array<DataVector, NeutrinoSpecies>>(
            DataVector{num_grid_points}, 0.0);
    for (size_t s = 0; s < NeutrinoSpecies; s++) {
      packet_energy_at_emission[s] = initial_packet_energy[s];
    }
    Initialization::mutate_assign<
        tmpl::list<Particles::MonteCarlo::Tags::DesiredPacketEnergyAtEmission<
            NeutrinoSpecies>>>(make_not_null(&box),
                               std::move(packet_energy_at_emission));

    // Initialize mortar data and coupling data.
    // Currently assumes a single neighbor on each face (i.e. no h-refinement)
    using MortarData =
        typename Particles::MonteCarlo::Tags::MortarDataTag<dim>::type;
    MortarData mortar_data;
    using CouplingData =
        typename Particles::MonteCarlo::Tags::GhostZoneCouplingDataTag<
            dim>::type;
    CouplingData coupling_data;
    const Element<dim>& element = db::get<::domain::Tags::Element<dim>>(box);
    for (const auto& [direction, neighbors] : element.neighbors()) {
      const size_t sliced_mesh_size =
          mesh.slice_away(direction.dimension()).number_of_grid_points();
      const DataVector zero_dv_slice(sliced_mesh_size, 0.0);
      const Index<dim - 1> sliced_mesh_extents =
          mesh.slice_away(direction.dimension()).extents();
      size_t sliced_mesh_size_with_ghost_zone = 1;
      for (size_t d = 0; d < dim - 1; d++) {
        sliced_mesh_size_with_ghost_zone *= ( sliced_mesh_extents[d]
                                             + 2 * num_ghost_zones );
      }
      const DataVector zero_dv_ghost_zones(sliced_mesh_size_with_ghost_zone,
                                           0.0);
      const tnsr::i<DataVector, dim, Frame::Inertial> zero_tnsr_ghost_zones =
          make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(
              zero_dv_ghost_zones, 0.0);

      for (const auto& neighbor : neighbors) {
        const DirectionalId<dim> mortar_id{direction, neighbor};
        mortar_data.rest_mass_density.emplace(mortar_id, zero_dv_slice);
        mortar_data.electron_fraction.emplace(mortar_id, zero_dv_slice);
        mortar_data.temperature.emplace(mortar_id, zero_dv_slice);
        mortar_data.cell_light_crossing_time.emplace(mortar_id, zero_dv_slice);
        coupling_data.coupling_tilde_tau.emplace(mortar_id,
                                                 zero_dv_ghost_zones);
        coupling_data.coupling_tilde_rho_ye.emplace(mortar_id,
                                                    zero_dv_ghost_zones);
        coupling_data.coupling_tilde_s.emplace(mortar_id,
                                               zero_tnsr_ghost_zones);
      }
    }
    Initialization::mutate_assign<
        tmpl::list<Particles::MonteCarlo::Tags::MortarDataTag<dim>>>(
        make_not_null(&box), std::move(mortar_data));
    Initialization::mutate_assign<
        tmpl::list<Particles::MonteCarlo::Tags::GhostZoneCouplingDataTag<dim>>>(
        make_not_null(&box), std::move(coupling_data));

    using GhostZoneData =
        typename Particles::MonteCarlo::Tags::McGhostZoneDataTag<dim>::type;
    GhostZoneData ghost_zone_data{};
    Initialization::mutate_assign<
        tmpl::list<Particles::MonteCarlo::Tags::McGhostZoneDataTag<dim>>>(
        make_not_null(&box), std::move(ghost_zone_data));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace Initialization::Actions
