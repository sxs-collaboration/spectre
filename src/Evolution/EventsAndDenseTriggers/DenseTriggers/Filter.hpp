// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>
#include <string>

#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
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
/// \snippet Test_Filter.cpp example
class Filter : public DenseTrigger {
 public:
  /// \cond
  Filter() = default;
  explicit Filter(CkMigrateMessage* const msg) noexcept : DenseTrigger(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Filter);  // NOLINT
  /// \endcond

  struct TriggerOption {
    static std::string name() noexcept { return "Trigger"; }
    using type = std::unique_ptr<DenseTrigger>;
    static constexpr Options::String help = "Dense trigger to filter";
  };

  struct FilterOption {
    static std::string name() noexcept { return "Filter"; }
    using type = std::unique_ptr<Trigger>;
    static constexpr Options::String help = "Non-dense trigger to filter with";
  };

  using options = tmpl::list<TriggerOption, FilterOption>;
  static constexpr Options::String help =
      "Filter activations of a dense trigger using a non-dense trigger.";

  explicit Filter(std::unique_ptr<DenseTrigger> trigger,
                  std::unique_ptr<Trigger> filter) noexcept;

  using is_triggered_argument_tags = tmpl::list<Tags::DataBox>;

  template <typename DbTags>
  Result is_triggered(const db::DataBox<DbTags>& box) const noexcept {
    auto result = trigger_->is_triggered(box);
    if (not filter_->is_triggered(box)) {
      result.is_triggered = false;
    }
    return result;
  }

  using is_ready_argument_tags = tmpl::list<Tags::DataBox>;

  template <typename Metavariables, typename ArrayIndex, typename Component,
            typename DbTags>
  bool is_ready(Parallel::GlobalCache<Metavariables>& cache,
                const ArrayIndex& array_index, const Component* const component,
                const db::DataBox<DbTags>& box) const noexcept {
    return trigger_->is_ready(box, cache, array_index, component);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept override;

 private:
  std::unique_ptr<DenseTrigger> trigger_{};
  std::unique_ptr<Trigger> filter_{};
};
}  // namespace DenseTriggers
