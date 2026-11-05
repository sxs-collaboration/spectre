// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateGetTypeAliasOrDefault.hpp"

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

namespace Actions {
namespace detail {
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(simple_tags)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(compute_tags)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(const_global_cache_tags)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(mutable_global_cache_tags)
}  // namespace detail
/*!
 * \ingroup ActionsGroup
 * \brief Apply the function `Mutator::apply` to the DataBox
 *
 * The function `Mutator::apply` is invoked with the `Mutator::argument_tags`.
 * The result of this computation is stored in the `Mutator::return_tags`.
 *
 * Uses:
 * - DataBox:
 *   - All elements in `Mutator::argument_tags`
 *
 * DataBox changes:
 * - Modifies:
 *   - All elements in `Mutator::return_tags`
 */
template <typename Mutator>
struct MutateApply {
  static std::string name() { return pretty_type::name<Mutator>(); }

  using simple_tags =
      detail::get_simple_tags_or_default_t<Mutator, tmpl::list<>>;
  using compute_tags =
      detail::get_compute_tags_or_default_t<Mutator, tmpl::list<>>;
  using const_global_cache_tags =
      detail::get_const_global_cache_tags_or_default_t<Mutator, tmpl::list<>>;
  using mutable_global_cache_tags =
      detail::get_mutable_global_cache_tags_or_default_t<Mutator, tmpl::list<>>;

  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      DataBox& box, const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate_apply<Mutator>(make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
