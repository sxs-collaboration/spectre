// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iomanip>
#include <sstream>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Parallel/ArrayCollection/IsDgElementCollection.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/OutputInbox.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/Actions/GetItemFromDistributedObject.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Utilities/TaggedTuple.hpp"

/*!
 * \brief Namespace for actions related to debugging deadlocks in communication.
 *
 * \details These actions will typically be run in the
 * `run_deadlock_analysis_simple_actions` function in the metavariables (if it
 * exists).
 */
namespace deadlock {
/*!
 * \brief Simple action that will print a variety of information on evolution DG
 * elements
 *
 * \details This will print the contents of the following inbox or DataBox tags:
 *
 * - `evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<3>`
 * - `evolution::dg::Tags::MortarNextTemporalId<3>`
 * - `evolution::dg::Tags::MortarDataHistory` (for LTS only)
 * - `evolution::dg::Tags::MortarData<3>` (for GTS only)
 *
 * Inbox tags are printed using the `Parallel::output_inbox` function. The
 * DataBox tags are printed with nice indenting for easy readability in the
 * stdout file.
 *
 * This can be generalized in the future to other dimensions if needed.
 */
struct PrintElementInfo {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index) {
    if constexpr (Parallel::is_dg_element_collection_v<ParallelComponent>) {
      auto& element =
          Parallel::local_synchronous_action<
              Parallel::Actions::GetItemFromDistributedOject<
                  typename ParallelComponent::element_collection_tag>>(
              Parallel::get_parallel_component<ParallelComponent>(cache))
              ->at(array_index);
      apply<ParallelComponent>(box, cache, array_index, &element);
    } else {
      apply<ParallelComponent>(
          box, cache, array_index,
          Parallel::local(Parallel::get_parallel_component<ParallelComponent>(
              cache)[array_index]));
    }
  }

 private:
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex, typename T>
  static void apply(db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& array_index, T* local_object_ptr) {
    auto& local_object = *local_object_ptr;

    const bool terminated = local_object.get_terminate();

    std::stringstream ss{};
    ss << std::scientific << std::setprecision(16);
    ss << "Element " << array_index
       << (terminated ? " terminated" : " did NOT terminate") << " at time "
       << db::get<::Tags::Time>(box) << ".";

    // Only print stuff if this element didn't terminate properly
    if (not terminated) {
      const std::string& next_action =
          local_object.deadlock_analysis_next_iterable_action();
      ss << " Next action: " << next_action << "\n";

      // Start with time step and next time
      const auto& step = db::get<::Tags::TimeStep>(box);
      // The time step only prints a slab (beginning/end) and a fraction so we
      // also print the approx numerical value of the step for easier reading
      ss << " Time step: " << step << ":" << step.value() << "\n";
      ss << " Next time: "
         << db::get<::Tags::Next<::Tags::TimeStepId>>(box).substep_time()
         << "\n";

      ss << " Inboxes:\n";

      const auto& inboxes = local_object.get_inboxes();

      const std::string mortar_inbox = Parallel::output_inbox<
          evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<3>>(inboxes,
                                                                        2_st);
      ss << mortar_inbox;

      ss << " Mortars:\n";

      const auto& mortar_next_temporal_id =
          db::get<evolution::dg::Tags::MortarNextTemporalId<3>>(box);

      ss << "  MortarNextTemporalId\n";
      for (const auto& [key, next_id] : mortar_next_temporal_id) {
        ss << "    Key: " << key << ", next time: " << next_id.substep_time()
           << "\n";
      }

      if constexpr (Metavariables::local_time_stepping) {
        const auto& mortar_data_history =
            db::get<evolution::dg::Tags::MortarDataHistory<
                3, typename db::add_tag_prefix<
                       ::Tags::dt,
                       typename Metavariables::system::variables_tag>::type>>(
                box);
        ss << "  MortarDataHistory:\n";

        for (const auto& [key, history] : mortar_data_history) {
          ss << "   Key: " << key << ", history:\n";
          history.template print<false>(ss, 4_st);
        }
      }
    } else {
      ss << "\n";
    }

    Parallel::printf("%s", ss.str());
  }
};
}  // namespace deadlock
