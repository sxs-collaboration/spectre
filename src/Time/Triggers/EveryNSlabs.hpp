// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <unordered_set>
#include <vector>

#include "Evolution/EventsAndTriggers/Trigger.hpp"
#include "Options/Options.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"

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
    using type = size_t;
    static constexpr OptionString help{"How frequently to trigger."};
    static type lower_bound() { return 1; }
  };
  struct Offset {
    using type = size_t;
    static constexpr OptionString help{"First slab to trigger on."};
    static type default_value() { return 0; }
  };

  using options = tmpl::list<N, Offset>;
  static constexpr OptionString help{
    "Trigger every N time slabs after a given offset."};

  EveryNSlabs(const size_t interval, const size_t offset) noexcept
      : interval_(interval), offset_(offset) {}

  using argument_tags = tmpl::list<Tags::TimeId>;

  bool operator()(const TimeId& time_id) const noexcept {
    const size_t slab_number = time_id.slab_number;
    return slab_number >= offset_ and (slab_number - offset_) % interval_ == 0;
  }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | interval_;
    p | offset_;
  }

 private:
  size_t interval_{0};
  size_t offset_{0};
};

/// \cond
template <typename KnownTriggers>
PUP::able::PUP_ID EveryNSlabs<KnownTriggers>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Triggers
