// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <pup.h>

#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic::Triggers {
/// \ingroup EventsAndTriggersGroup
/// Trigger every N iterations of the solver identifid by the `Label`, after a
/// given offset.
template <typename Label>
class EveryNIterations : public Trigger {
 public:
  /// \cond
  EveryNIterations() = default;
  explicit EveryNIterations(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(EveryNIterations);  // NOLINT
  /// \endcond

  struct N {
    using type = uint64_t;
    static constexpr Options::String help{"How frequently to trigger."};
    static type lower_bound() { return 1; }
  };
  struct Offset {
    using type = uint64_t;
    static constexpr Options::String help{"First iteration to trigger on."};
  };

  using options = tmpl::list<N, Offset>;
  static constexpr Options::String help{
      "Trigger every N iterations after a given offset."};

  EveryNIterations(const uint64_t interval, const uint64_t offset)
      : interval_(interval), offset_(offset) {}

  using argument_tags = tmpl::list<Convergence::Tags::IterationId<Label>>;

  bool operator()(const size_t iteration_id) const {
    const auto step_number = static_cast<uint64_t>(iteration_id);
    return step_number >= offset_ and (step_number - offset_) % interval_ == 0;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) {
    p | interval_;
    p | offset_;
  }

 private:
  uint64_t interval_{0};
  uint64_t offset_{0};
};

/// \cond
template <typename Label>
PUP::able::PUP_ID EveryNIterations<Label>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace elliptic::Triggers
