// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
struct DataBox;
}  // namespace Tags
namespace db {
template <typename TagsList> class DataBox;
}  // namespace db
/// \endcond

namespace DenseTriggers {
/// \ingroup EventsAndTriggersGroup
/// %Filter activations of a dense trigger using a non-dense trigger.
///
/// For example, to trigger every 10 starting at 100, one could use
///
/// \snippet DenseTriggers/Test_Filter.cpp example
class Filter : public DenseTrigger {
 public:
  /// \cond
  Filter() = default;
  explicit Filter(CkMigrateMessage* const msg) : DenseTrigger(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Filter);  // NOLINT
  /// \endcond

  struct TriggerOption {
    static std::string name() { return "Trigger"; }
    using type = std::unique_ptr<DenseTrigger>;
    static constexpr Options::String help = "Dense trigger to filter";
  };

  struct FilterOption {
    static std::string name() { return "Filter"; }
    using type = std::unique_ptr<Trigger>;
    static constexpr Options::String help = "Non-dense trigger to filter with";
  };

  using options = tmpl::list<TriggerOption, FilterOption>;
  static constexpr Options::String help =
      "Filter activations of a dense trigger using a non-dense trigger.";

  explicit Filter(std::unique_ptr<DenseTrigger> trigger,
                  std::unique_ptr<Trigger> filter);

  using is_triggered_argument_tags = tmpl::list<Tags::DataBox>;

  template <typename Metavariables, typename ArrayIndex, typename Component,
            typename DbTags>
  std::optional<bool> is_triggered(Parallel::GlobalCache<Metavariables>& cache,
                                   const ArrayIndex& array_index,
                                   const Component* component,
                                   const db::DataBox<DbTags>& box) const {
    auto result = trigger_->is_triggered(box, cache, array_index, component);
    if (result == std::optional{true}) {
      return filter_->is_triggered(box);
    }
    return result;
  }

  using next_check_time_argument_tags = tmpl::list<Tags::DataBox>;

  template <typename Metavariables, typename ArrayIndex, typename Component,
            typename DbTags>
  std::optional<double> next_check_time(
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const Component* component,
      const db::DataBox<DbTags>& box) const {
    return trigger_->next_check_time(box, cache, array_index, component);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  std::unique_ptr<DenseTrigger> trigger_{};
  std::unique_ptr<Trigger> filter_{};
};
}  // namespace DenseTriggers
