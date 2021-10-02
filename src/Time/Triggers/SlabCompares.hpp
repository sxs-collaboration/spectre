// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <pup.h>

#include "Options/Comparator.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct TimeStepId;
}  // namespace Tags
/// \endcond

namespace Triggers {
/// \ingroup EventsAndTriggersGroup
/// \ingroup TimeGroup
/// Trigger based on a comparison with the slab number.
class SlabCompares : public Trigger {
 public:
  /// \cond
  SlabCompares() = default;
  explicit SlabCompares(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(SlabCompares);  // NOLINT
  /// \endcond

  struct Comparison {
    using type = Options::Comparator;
    constexpr static Options::String help = "Comparison type";
  };

  struct Value {
    using type = uint64_t;
    constexpr static Options::String help = "Value to compare to";
  };

  using options = tmpl::list<Comparison, Value>;
  static constexpr Options::String help{
      "Trigger based on a comparison with the slab number."};

  explicit SlabCompares(const Options::Comparator comparator,
                        const uint64_t slab_number)
      : comparator_(comparator), slab_number_(slab_number) {}

  using argument_tags = tmpl::list<Tags::TimeStepId>;

  bool operator()(const TimeStepId& time_step_id) const {
    return comparator_(static_cast<uint64_t>(time_step_id.slab_number()),
                       slab_number_);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) {
    p | comparator_;
    p | slab_number_;
  }

 private:
  Options::Comparator comparator_{};
  uint64_t slab_number_{};
};
}  // namespace Triggers
