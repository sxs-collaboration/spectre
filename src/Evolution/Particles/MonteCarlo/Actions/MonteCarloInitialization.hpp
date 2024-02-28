// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "Evolution/Particles/MonteCarlo/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
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
/// - DataBox:
///   * `Tags::Mesh<3>`
///
/// DataBox changes:
/// - Adds:
///   * Particles::MonteCarlo::Tags::McPacketsOnElement
///   * Particles::MonteCarlo::Tags::EmissionInCell
///
/// - Removes: nothing
/// - Modifies: nothing
template <size_t EnergyBins, size_t NeutrinoSpecies>
struct MonteCarlo {
 public:
  using simple_tags =
      tmpl::list<Particles::MonteCarlo::Tags::McPacketsOnElement,
                 Particles::MonteCarlo::Tags::EmissionInCell<
                     DataVector, EnergyBins, NeutrinoSpecies>>;

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
    typename Particles::MonteCarlo::Tags::McPacketsOnElement::type all_packets;
    Initialization::mutate_assign<
        tmpl::list<Particles::MonteCarlo::Tags::McPacketsOnElement>>(
        make_not_null(&box), std::move(all_packets));

    const size_t num_grid_points =
        db::get<domain::Tags::Mesh<3>>(box).number_of_grid_points();
    using emission_tag = typename Particles::MonteCarlo::Tags::EmissionInCell<
        DataVector, EnergyBins, NeutrinoSpecies>;
    typename emission_tag::type emission_in_cell;
    for (size_t n = 0; n < EnergyBins; n++) {
      emission_in_cell[n].fill(DataVector(num_grid_points, 0.0));
    }
    Initialization::mutate_assign<
        tmpl::list<Particles::MonteCarlo::Tags::EmissionInCell<
            DataVector, EnergyBins, NeutrinoSpecies>>>(
        make_not_null(&box), std::move(emission_in_cell));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace Initialization::Actions
