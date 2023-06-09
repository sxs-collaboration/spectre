// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <pup.h>
#include <vector>

#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Options/String.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
struct DataBox;
struct TimeStepId;
}  // namespace Tags
namespace db {
template <typename TagsList> class DataBox;
}  // namespace db
/// \endcond

namespace DenseTriggers {
/// \ingroup EventsAndTriggersGroup
/// Trigger when any of a collection of DenseTriggers triggers.
class Or : public DenseTrigger {
 public:
  /// \cond
  Or() = default;
  explicit Or(CkMigrateMessage* const msg) : DenseTrigger(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Or);  // NOLINT
  /// \endcond

  static constexpr Options::String help =
      "Trigger when any of a collection of triggers triggers.";

  explicit Or(std::vector<std::unique_ptr<DenseTrigger>> triggers);

  using is_triggered_argument_tags = tmpl::list<Tags::DataBox>;

  template <typename Metavariables, typename ArrayIndex, typename Component,
            typename DbTags>
  std::optional<bool> is_triggered(Parallel::GlobalCache<Metavariables>& cache,
                                   const ArrayIndex& array_index,
                                   const Component* component,
                                   const db::DataBox<DbTags>& box) const {
    bool is_ready = true;
    for (const auto& trigger : triggers_) {
      const auto sub_result =
          trigger->is_triggered(box, cache, array_index, component);
      if (sub_result.has_value()) {
        if (*sub_result) {
          // No need to wait for all the other triggers to be ready.
          return true;
        }
      } else {
        is_ready = false;
      }
    }
    return is_ready ? std::optional{false} : std::nullopt;
  }

  using next_check_time_argument_tags =
      tmpl::list<Tags::TimeStepId, Tags::DataBox>;

  template <typename Metavariables, typename ArrayIndex, typename Component,
            typename DbTags>
  std::optional<double> next_check_time(
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const Component* component,
      const TimeStepId& time_step_id, const db::DataBox<DbTags>& box) const {
    const evolution_less<double> before{time_step_id.time_runs_forward()};
    double result = before.infinity();
    for (const auto& trigger : triggers_) {
      const auto sub_result =
          trigger->next_check_time(box, cache, array_index, component);
      if (not sub_result.has_value()) {
        return std::nullopt;
      }
      result = std::min(*sub_result, result, before);
    }
    return result;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  std::vector<std::unique_ptr<DenseTrigger>> triggers_{};
};
}  // namespace DenseTriggers

template <>
struct Options::create_from_yaml<DenseTriggers::Or> {
  template <typename Metavariables>
  static DenseTriggers::Or create(const Option& options) {
    return DenseTriggers::Or(
        options.parse_as<std::vector<std::unique_ptr<DenseTrigger>>,
                         Metavariables>());
  }
};
