// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <pup.h>

#include "Evolution/EventsAndTriggers/Trigger.hpp"
#include "Time/TimeId.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct TimeId;
}  // namespace Tags
/// \endcond

namespace Triggers {
/// \ingroup EventsAndTriggersGroup
/// \ingroup TimeGroup
/// Trigger every N time slabs after a given offset.
template <typename KnownTriggers>
class EveryNSlabs : public Trigger<KnownTriggers> {
 public:
  /// \cond
  EveryNSlabs() = default;
  explicit EveryNSlabs(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(EveryNSlabs);  // NOLINT
  /// \endcond

  struct N {
    using type = uint64_t;
    static constexpr OptionString help{"How frequently to trigger."};
    static type lower_bound() { return 1; }
  };
  struct Offset {
    using type = uint64_t;
    static constexpr OptionString help{"First slab to trigger on."};
    static type default_value() { return 0; }
  };

  using options = tmpl::list<N, Offset>;
  static constexpr OptionString help{
    "Trigger every N time slabs after a given offset."};

  EveryNSlabs(const uint64_t interval, const uint64_t offset) noexcept
      : interval_(interval), offset_(offset) {}

  using argument_tags = tmpl::list<Tags::TimeId>;

  bool operator()(const TimeId& time_id) const noexcept {
    const auto slab_number = static_cast<uint64_t>(time_id.slab_number());
    return slab_number >= offset_ and (slab_number - offset_) % interval_ == 0;
  }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | interval_;
    p | offset_;
  }

 private:
  uint64_t interval_{0};
  uint64_t offset_{0};
};

/// \cond
template <typename KnownTriggers>
PUP::able::PUP_ID EveryNSlabs<KnownTriggers>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Triggers
