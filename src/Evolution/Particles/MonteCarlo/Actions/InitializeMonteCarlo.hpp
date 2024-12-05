// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <random>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Evolution/Particles/MonteCarlo/GhostZoneCommunicationTags.hpp"
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
///
/// DataBox changes:
/// - Adds:
///   * Particles::MonteCarlo::Tags::PacketsOnElement
///   * Particles::MonteCarlo::Tags::RandomNumberGenerator
///   * Particles::MonteCarlo::Tags::DesiredPacketEnergyAtEmission<
///                                  NeutrinoSpecies>
///   * Particles::MonteCarlo::Tags::CellLightCrossingTime<DataVector>
///   * Background hydro variables
///   * Particles::MonteCarlo::Tags::MortarDataTag<dim>
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
                 Particles::MonteCarlo::Tags::CellLightCrossingTime<DataVector>,
                 hydro_variables_tag,
                 Particles::MonteCarlo::Tags::MortarDataTag<dim>,
                 Particles::MonteCarlo::Tags::McGhostZoneDataTag<dim>>;

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
    const size_t num_grid_points =
        db::get<evolution::dg::subcell::Tags::Mesh<dim>>(box)
            .number_of_grid_points();
    using derived_classes =
        tmpl::at<typename Metavariables::factory_creation::factory_classes,
                 evolution::initial_data::InitialData>;
    using HydroVars = typename hydro_variables_tag::type;
    call_with_dynamic_type<void, derived_classes>(
        &db::get<evolution::initial_data::Tags::InitialData>(box),
        [&box](const auto* const data_or_solution) {
          static constexpr size_t dim = System::volume_dim;
          const double initial_time = db::get<::Tags::Time>(box);
          const size_t num_grid_points =
              db::get<evolution::dg::subcell::Tags::Mesh<dim>>(box)
                  .number_of_grid_points();
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

    typename Particles::MonteCarlo::Tags::PacketsOnElement::type all_packets;
    Initialization::mutate_assign<
        tmpl::list<Particles::MonteCarlo::Tags::PacketsOnElement>>(
        make_not_null(&box), std::move(all_packets));

    const unsigned long seed =
        std::random_device{}();  // static_cast<unsigned long>(time(NULL));
    typename Particles::MonteCarlo::Tags::RandomNumberGenerator::type rng(seed);

    Initialization::mutate_assign<
        tmpl::list<Particles::MonteCarlo::Tags::RandomNumberGenerator>>(
        make_not_null(&box), std::move(rng));

    // Currently hard-code energy at emission; should be set by option
    typename Particles::MonteCarlo::Tags::DesiredPacketEnergyAtEmission<
        NeutrinoSpecies>::type packet_energy_at_emission =
        make_with_value<std::array<DataVector, NeutrinoSpecies>>(
            DataVector{num_grid_points}, 1.e-12);
    Initialization::mutate_assign<
        tmpl::list<Particles::MonteCarlo::Tags::DesiredPacketEnergyAtEmission<
            NeutrinoSpecies>>>(make_not_null(&box),
                               std::move(packet_energy_at_emission));

    typename Particles::MonteCarlo::Tags::CellLightCrossingTime<
        DataVector>::type cell_light_crossing_time(num_grid_points, 1.0);
    Initialization::mutate_assign<tmpl::list<
        Particles::MonteCarlo::Tags::CellLightCrossingTime<DataVector>>>(
        make_not_null(&box), std::move(cell_light_crossing_time));

    // Initialize empty mortar data; do we need more at initialization stage?
    using MortarData =
        typename Particles::MonteCarlo::Tags::MortarDataTag<dim>::type;
    MortarData mortar_data;
    Initialization::mutate_assign<
        tmpl::list<Particles::MonteCarlo::Tags::MortarDataTag<dim>>>(
        make_not_null(&box), std::move(mortar_data));

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
