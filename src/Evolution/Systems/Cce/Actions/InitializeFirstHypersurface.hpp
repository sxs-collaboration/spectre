// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/InitializeCce.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/ScriPlusValues.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace Actions {

/*!
 * \ingroup Actions
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
 */
struct InitializeFirstHypersurface {
  using const_global_cache_tags =
      tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints, Tags::InitializeJ>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate_apply<InitializeJ::mutate_tags, InitializeJ::argument_tags>(
        db::get<Tags::InitializeJ>(box), make_not_null(&box));
    db::mutate_apply<InitializeScriPlusValue<Tags::InertialRetardedTime>>(
        make_not_null(&box));
    return {std::move(box)};
  }
};
}  // namespace Actions
}  // namespace Cce
