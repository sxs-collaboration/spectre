// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/ScriPlusValues.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Given initial boundary data for \f$J\f$ and \f$\partial_r J\f$,
 * computes the initial hypersurface quantities \f$J\f$ and gauge values.
 *
 * \details This action is to be called after boundary data has been received,
 * but before the time-stepping evolution loop. So, it should be either late in
 * an initialization phase or early (before a `Actions::Goto` loop or similar)
 * in the `Evolve` phase.
 *
 * Internally, this dispatches to the call function of
 * `Tags::InitializeJ`, which designates a hypersurface initial data generator
 * chosen by input file options, `InitializeGauge`, and
 * `InitializeScriPlusValue<Tags::InertialRetardedTime>` to perform the
 * computations. Refer to the documentation for those mutators for mathematical
 * details.
 *
 * \note This action accesses the base tag `Cce::Tags::InitializeJBase`,
 * trusting that a tag that inherits from that base tag is present in the box or
 * the global cache. Typically, this tag should be added by the worldtube
 * boundary component, as the type of initial data is decided by the type of the
 * worldtube boundary data.
 */
template <bool uses_partially_flat_cartesian_coordinates>
struct InitializeFirstHypersurface {
  using const_global_cache_tags =
      tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // In some contexts, this action may get re-run (e.g. self-start procedure)
    // In those cases, we do not want to alter the existing hypersurface data,
    // so we just exit. However, we do want to re-run the action each time
    // the self start 'reset's from the beginning
    if (db::get<::Tags::TimeStepId>(box).slab_number() > 0 or
        db::get<::Tags::TimeStepId>(box).substep_time().fraction() != 0) {
      return {std::move(box)};
    }
    db::mutate_apply<
        typename InitializeJ::InitializeJ<
            uses_partially_flat_cartesian_coordinates>::mutate_tags,
        typename InitializeJ::InitializeJ<
            uses_partially_flat_cartesian_coordinates>::argument_tags>(
        db::get<Tags::InitializeJBase>(box), make_not_null(&box));
    db::mutate_apply<InitializeScriPlusValue<Tags::InertialRetardedTime>>(
        make_not_null(&box),
        db::get<::Tags::TimeStepId>(box).substep_time().value());
    return {std::move(box)};
  }
};
}  // namespace Actions
}  // namespace Cce
