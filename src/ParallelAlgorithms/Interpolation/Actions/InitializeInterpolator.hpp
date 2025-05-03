// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond

namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace intrp::Tags {
template <size_t Dim>
struct NumberOfElements;
template <typename Metavariables>
struct InterpolatedVarsHolders;
template <typename Metavariables, typename TemporalId>
struct VolumeVarsInfo;
struct Verbosity;
}  // namespace intrp::Tags
/// \endcond

namespace intrp::Actions {

/// \ingroup ActionsGroup
/// \brief Initializes an Interpolator
///
/// Uses: nothing
///
/// DataBox changes:
/// - Adds:
///   - `Tags::NumberOfElements`
///   - each tag in the template argument VolumeVarsInfos, which may either be a
///     single `Tags::VolumeVarsInfo<Metavariables, TemporalId>` or a
///     `tmpl::list` of multiple tags for `VolumeVarsInfo`.
///   - `Tags::InterpolatedVarsHolders<Metavariables>`
/// - Removes: nothing
/// - Modifies: nothing
template <size_t Dim, typename VolumeVarsInfos,
          typename InterpolatedVarsHolders>
struct InitializeInterpolator {
  using return_tag_list =
      tmpl::flatten<tmpl::list<Tags::NumberOfElements<Dim>, VolumeVarsInfos,
                               InterpolatedVarsHolders>>;

  using simple_tags = return_tag_list;
  using compute_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<intrp::Tags::Verbosity>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    using sequential_targets =
        intrp::InterpolationTarget_detail::get_sequential_target_tags<
            Metavariables>;
    db::mutate<Tags::NumberOfElements<Dim>>(
        [](const gsl::not_null<std::unordered_map<
               std::string, std::unordered_set<ElementId<Dim>>>*>
               num_elements) {
          tmpl::for_each<sequential_targets>([&](auto target_v) {
            using target = tmpl::type_from<decltype(target_v)>;
            const std::string& target_name = pretty_type::name<target>();

            (*num_elements)[target_name] = std::unordered_set<ElementId<Dim>>{};
          });
        },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace intrp::Actions
