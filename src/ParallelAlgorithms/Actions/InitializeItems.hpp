// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
/// \endcond

namespace Initialization {
namespace Actions {
/*!
 * \ingroup ActionsGroup
 * \brief Mutate DataBox items by calling db::mutate_apply on each Mutator in
 *  the order they are specified
 *
 * There's a specialization for `InitializeItems<tmpl::list<Mutators...>>` that
 * can also be used if a `tmpl::list` is available.
 *
 * \details In addition to the requirements specified by db::mutate_apply, each
 * Mutator must define the type aliases of this action.
 */
template <typename... Mutators>
struct InitializeItems {
  /// Tags for constant items added to the GlobalCache.  These items are
  /// initialized from input file options.
  using const_global_cache_tags = tmpl::remove_duplicates<tmpl::flatten<
      tmpl::append<typename Mutators::const_global_cache_tags...>>>;
  /// Tags for mutable items added to the MutableGlobalCache.  These items are
  /// initialized from input file options.
  using mutable_global_cache_tags = tmpl::remove_duplicates<tmpl::flatten<
      tmpl::append<typename Mutators::mutable_global_cache_tags...>>>;
  /// Tags for simple DataBox items that are initialized from input file options
  using simple_tags_from_options = tmpl::remove_duplicates<tmpl::flatten<
      tmpl::append<typename Mutators::simple_tags_from_options...>>>;
  /// Tags for simple DataBox items that are default initialized.  They may be
  /// mutated by the Mutators.
  using simple_tags = tmpl::remove_duplicates<
      tmpl::flatten<tmpl::append<typename Mutators::simple_tags...>>>;
  /// Tags for immutable DataBox items (compute items or reference items).
  using compute_tags = tmpl::remove_duplicates<
      tmpl::flatten<tmpl::append<typename Mutators::compute_tags...>>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    EXPAND_PACK_LEFT_TO_RIGHT(db::mutate_apply<Mutators>(make_not_null(&box)));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/// \cond
template <typename... Mutators>
struct InitializeItems<tmpl::list<Mutators...>>
    : public InitializeItems<Mutators...> {};
/// \endcond
}  // namespace Actions
}  // namespace Initialization
