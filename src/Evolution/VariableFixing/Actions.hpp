// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/VariableFixing/RadiallyFallingFloor.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples

namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
/// \endcond

namespace Actions {
/// \ingroup ActionsGroup
/// \brief Fix the pressure and rest mass density if they are too low.
///
/// Uses:
/// - DataBox:
///   - VariableFixer::return_tags
///   - VariableFixer::fixing_scheme::argument_tags
/// - ConstGlobalCache:
///   - VariableFixer::const_global_cache_tags
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: VariableFixer::return_tags
template <typename VariableFixer>
struct ApplyVariableFixer {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::size<DbTagsList>::value != 0> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    helper(box, cache, typename VariableFixer::const_global_cache_tag_list{});
    return std::forward_as_tuple(std::move(box));
  }

 private:
  template <typename DbTagsList, typename Metavariables,
            typename... MyCacheTags>
  static void helper(db::DataBox<DbTagsList>& box,
                     const Parallel::ConstGlobalCache<Metavariables>& cache,
                     tmpl::list<MyCacheTags...> /*meta*/) {
    db::mutate_apply<typename VariableFixer::return_tags,
                     typename VariableFixer::argument_tags>(
        VariableFixer{}, make_not_null(&box),
        Parallel::get<MyCacheTags>(cache)...);
  }
};
}  // namespace Actions
