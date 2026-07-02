// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/Tags/ChangeFixedLtsRatioTags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class TimeStepId;
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
struct FixedLtsRatio;
struct StepNumberWithinSlab;
struct TimeStepId;
}  // namespace Tags
namespace tuples {
template <class... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace evolution::dg::Actions {
/// Adjust Tags::FixedLtsRatio based on previous executions of
/// Events::ChangeFixedLtsRatio.
struct ChangeFixedLtsRatio {
  using simple_tags =
      tmpl::list<Tags::ChangeFixedLtsRatio::NumberOfExpectedMessages,
                 Tags::ChangeFixedLtsRatio::NewStepSize>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if (db::mutate_apply<Impl>(make_not_null(&box))) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    } else {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }
  }

 private:
  struct Impl {
    using return_tags =
        tmpl::list<::Tags::FixedLtsRatio,
                   Tags::ChangeFixedLtsRatio::NumberOfExpectedMessages,
                   Tags::ChangeFixedLtsRatio::NewStepSize>;
    using argument_tags =
        tmpl::list<::Tags::TimeStepId, ::Tags::StepNumberWithinSlab>;

    using StepId = std::pair<int64_t, uint64_t>;

    static bool apply(
        gsl::not_null<std::optional<size_t>*> fixed_lts_ratio,
        gsl::not_null<std::map<StepId, size_t>*> expected_messages_map,
        gsl::not_null<std::map<StepId, std::vector<double>>*>
            new_step_size_messages_map,
        const TimeStepId& time_step_id, uint64_t step_number_within_slab);
  };
};
}  // namespace evolution::dg::Actions
