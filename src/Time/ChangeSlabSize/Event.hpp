// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/ChangeSlabSize/Tags.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Time/TimeStepRequestProcessor.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct DataBox;
struct TimeStepId;
}  // namespace Tags
/// \endcond

namespace Events {
namespace ChangeSlabSize_detail {
struct StoreNewSlabSize {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const int64_t slab_number,
                    const TimeStepRequestProcessor& requests) {
    db::mutate<::Tags::ChangeSlabSize::NewSlabSize>(
        [&](const gsl::not_null<
            std::map<int64_t, std::vector<TimeStepRequestProcessor>>*>
                sizes) { (*sizes)[slab_number].emplace_back(requests); },
        make_not_null(&box));
  }
};
}  // namespace ChangeSlabSize_detail

/// \ingroup TimeGroup
/// %Trigger a slab size change.
///
/// The new size will be the minimum suggested by any of the provided
/// step choosers on any element.  This requires a global reduction,
/// so it is possible to delay the change until a later slab to avoid
/// a global synchronization.  The actual change is carried out by
/// Actions::ChangeSlabSize.
///
/// When running with global time-stepping, the slab size and step
/// size are the same, so this adjusts the step size used by the time
/// integration.  With local time-stepping this controls the interval
/// between times when the sequences of steps on all elements are
/// forced to align.
class ChangeSlabSize : public Event {
  using ReductionData = Parallel::ReductionData<
      Parallel::ReductionDatum<int64_t, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<TimeStepRequestProcessor, funcl::Plus<>>>;

 public:
  /// \cond
  explicit ChangeSlabSize(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ChangeSlabSize);  // NOLINT
  /// \endcond

  struct StepChoosers {
    static constexpr Options::String help = "Limits on slab size";
    using type =
        std::vector<std::unique_ptr<StepChooser<StepChooserUse::Slab>>>;
    static size_t lower_bound_on_size() { return 1; }
  };

  struct DelayChange {
    static constexpr Options::String help = "Slabs to wait before changing";
    using type = uint64_t;
  };

  using options = tmpl::list<StepChoosers, DelayChange>;
  static constexpr Options::String help =
      "Trigger a slab size change chosen by the provided step choosers.\n"
      "The actual changing of the slab size can be delayed until a later\n"
      "slab to improve parallelization.";

  ChangeSlabSize() = default;
  ChangeSlabSize(std::vector<std::unique_ptr<StepChooser<StepChooserUse::Slab>>>
                     step_choosers,
                 const uint64_t delay_change, const Options::Context& context)
      : step_choosers_(std::move(step_choosers)), delay_change_(delay_change) {
    if (delay_change != 0) {
      for (const auto& chooser : step_choosers_) {
        if (not chooser->can_be_delayed()) {
          // The runtime name might not be exactly the same as the one
          // used by the factory, but hopefully it's close enough that
          // the user can figure it out.
          PARSE_ERROR(context,
                      "The " << pretty_type::get_runtime_type_name(*chooser)
                      << " StepChooser cannot be applied with a delay.");
        }
      }
    }
  }

  using compute_tags_for_observation_box = tmpl::list<>;

  // Need a const version of the full box for the step choosers, but
  // can't get a const version while mutating other tags, so request a
  // mutable version.
  using return_tags = tmpl::list<::Tags::DataBox>;
  using argument_tags = tmpl::list<::Tags::TimeStepId>;

  template <typename DbTags, typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  void operator()(const gsl::not_null<db::DataBox<DbTags>*> box,
                  const TimeStepId& time_step_id,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const ParallelComponent* const /*meta*/,
                  const ObservationValue& /*observation_value*/) const {
    const auto next_changable_slab = time_step_id.is_at_slab_boundary()
                                         ? time_step_id.slab_number()
                                         : time_step_id.slab_number() + 1;
    const auto slab_to_change =
        next_changable_slab + static_cast<int64_t>(delay_change_);
    const auto slab_start = time_step_id.step_time();
    const auto current_slab_size = (time_step_id.time_runs_forward() ? 1 : -1) *
                                   slab_start.slab().duration();

    TimeStepRequestProcessor step_requests(time_step_id.time_runs_forward());
    bool synchronization_required = false;
    for (const auto& step_chooser : step_choosers_) {
      step_requests.process(
          step_chooser->desired_step(current_slab_size.value(), *box).first);
      // We must synchronize if any step chooser requires it, not just
      // the limiting one, because choosers requiring synchronization
      // can be limiting on some processors and not others.
      if (not synchronization_required) {
        synchronization_required = step_chooser->uses_local_data();
      }
    }

    db::mutate<::Tags::ChangeSlabSize::NumberOfExpectedMessages>(
        [&](const gsl::not_null<std::map<int64_t, size_t>*> expected) {
          ++(*expected)[slab_to_change];
        },
        box);

    const auto& component_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    const auto& self_proxy = component_proxy[array_index];
    if (synchronization_required) {
      Parallel::contribute_to_reduction<
          ChangeSlabSize_detail::StoreNewSlabSize>(
          ReductionData(slab_to_change, step_requests), self_proxy,
          component_proxy);
    } else {
      db::mutate<::Tags::ChangeSlabSize::NewSlabSize>(
          [&](const gsl::not_null<
              std::map<int64_t, std::vector<TimeStepRequestProcessor>>*>
                  sizes) { (*sizes)[slab_to_change].push_back(step_requests); },
          box);
    }
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override {
    // This depends on the chosen StepChoosers, but they don't have a
    // way to report this information so we just return true to be
    // safe.
    return true;
  }

  template <typename F>
  void for_each_step_chooser(F&& f) const {
    for (const auto& step_chooser : step_choosers_) {
      f(*step_chooser);
    }
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    Event::pup(p);
    p | step_choosers_;
    p | delay_change_;
  }

 private:
  std::vector<std::unique_ptr<StepChooser<StepChooserUse::Slab>>>
      step_choosers_;
  uint64_t delay_change_ = std::numeric_limits<uint64_t>::max();
};
}  // namespace Events
