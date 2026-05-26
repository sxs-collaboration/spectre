// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/EqualRateRegions.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Options/String.hpp"
#include "Parallel/ArrayCollection/IsDgElementCollection.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/CollectStepperErrorTolerances.hpp"
#include "Time/RequestsStepperErrorTolerances.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/TimeStepRequestProcessor.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel::Tags {
template <typename ParallelComponent, typename SectionIdTag>
struct Section;
}  // namespace Parallel::Tags
namespace Tags {
struct DataBox;
struct StepNumberWithinSlab;
struct TimeStep;
struct TimeStepId;
}  // namespace Tags
namespace domain::Tags {
template <size_t VolumeDim>
struct Element;
}  // namespace domain::Tags
namespace evolution::dg::Tags {
struct EqualRateRegionId;
template <size_t Dim>
struct EqualRateRegions;
}  // namespace evolution::dg::Tags
namespace evolution::dg::Tags::ChangeFixedLtsRatio {
struct NewStepSize;
struct NumberOfExpectedMessages;
}  //  namespace evolution::dg::Tags::ChangeFixedLtsRatio
/// \endcond

namespace dg::Events {
namespace ChangeFixedLtsRatio_detail {
template <size_t Dim>
struct StoreNewStep {
  using StepId = std::pair<int64_t, uint64_t>;

  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const evolution::dg::EqualRateRegionId& region,
                    const StepId& step_to_change, const double new_step_size) {
    const auto& equal_rate_regions =
        db::get<evolution::dg::Tags::EqualRateRegions<Dim>>(box);
    const auto& element = db::get<domain::Tags::Element<Dim>>(box);
    if (equal_rate_regions.is_in_region(region, element.id())) {
      db::mutate<evolution::dg::Tags::ChangeFixedLtsRatio::NewStepSize>(
          [&](const gsl::not_null<std::map<StepId, std::vector<double>>*>
                  step_sizes) {
            (*step_sizes)[step_to_change].push_back(new_step_size);
          },
          make_not_null(&box));
    }
  }
};
}  // namespace ChangeFixedLtsRatio_detail

template <size_t Dim, typename StepChoosersToUse = AllStepChoosers>
class ChangeFixedLtsRatio : public Event,
                            public RequestsStepperErrorTolerances {
  using StepId = std::pair<int64_t, uint64_t>;

  using ReductionData = Parallel::ReductionData<
      Parallel::ReductionDatum<evolution::dg::EqualRateRegionId,
                               funcl::AssertEqual<>>,
      Parallel::ReductionDatum<StepId, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<double, funcl::Min<>>>;

 public:
  ChangeFixedLtsRatio() = default;

  /// \cond
  explicit ChangeFixedLtsRatio(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ChangeFixedLtsRatio);  // NOLINT
  /// \endcond

  struct StepChoosers {
    static constexpr Options::String help = "Limits on step size";
    using type =
        std::vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>;
    static size_t lower_bound_on_size() { return 1; }
  };

  struct DelayChange {
    static constexpr Options::String help = "Steps to wait before changing";
    using type = uint64_t;
  };

  struct Regions {
    using type =
        Options::Auto<std::vector<std::string>, Options::AutoLabel::All>;
    static constexpr Options::String help{
        "Regions to adjust.  Adjustments for each region are independent."};
  };

  using options = tmpl::list<StepChoosers, DelayChange, Regions>;
  static constexpr Options::String help =
      "Change the number of steps per slab in regions with synchronized steps.";

  ChangeFixedLtsRatio(
      std::vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>
          step_choosers,
      const uint64_t delay_change,
      std::optional<std::vector<std::string>> regions,
      const Options::Context& context = {})
      : step_choosers_(std::move(step_choosers)),
        delay_change_(delay_change),
        regions_(std::move(regions)) {
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
  using argument_tags = tmpl::list<>;

  template <typename DbTags, typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  void operator()(const gsl::not_null<db::DataBox<DbTags>*> box,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const ParallelComponent* const /*meta*/,
                  const ObservationValue& /*observation_value*/) const {
    const auto& time_step_id = db::get<::Tags::TimeStepId>(*box);
    if (time_step_id.substep() != 0) {
      ERROR("Changing step ratio on a substep not implemented.");
    }

    auto& section = db::get_mutable_reference<Parallel::Tags::Section<
        ParallelComponent, evolution::dg::Tags::EqualRateRegionId>>(box);

    if (not section.has_value()) {
      return;
    }

    if (regions_.has_value()) {
      const auto& equal_rate_regions =
          db::get<evolution::dg::Tags::EqualRateRegions<Dim>>(*box);
      const auto& region_map = equal_rate_regions.regions();
      bool active = false;
      // Iterate over the regions instead of looking up our region's
      // name so we catch typos in the input file option.
      for (const auto& region_name : *regions_) {
        const auto region_it = region_map.find(region_name);
        if (region_it == region_map.end()) {
          ERROR_NO_TRACE("Unknown region name in ChangeFixedLtsRatio: "
                         << region_name
                         << ".  Known regions: " << keys_of(region_map));
        }
        if (region_it->second == section->id()) {
          active = true;
          // Don't break here so we check the remaining names for validity.
        }
      }
      if (not active) {
        return;
      }
    }

    const double current_step = db::get<::Tags::TimeStep>(*box).value();

    TimeStepRequestProcessor step_requests(time_step_id.time_runs_forward());
    bool synchronization_required = false;
    for (const auto& step_chooser : step_choosers_) {
      step_requests.process(
          step_chooser->template desired_step<StepChoosersToUse>(current_step,
                                                                 *box));
      // We must synchronize if any step chooser requires it, not just
      // the limiting one, because choosers requiring synchronization
      // can be limiting on some processors and not others.
      if (not synchronization_required) {
        synchronization_required = step_chooser->uses_local_data();
      }
    }

    const double desired_step_size = std::abs(step_requests.step_size(
        time_step_id.step_time().value(), current_step));

    // The processing action will consider anything past the end of
    // the slab as at the start of the next slab.  The delay could be
    // off by a bit because of that, but we don't have to worry about
    // figuring out when the slab will end.
    const StepId step_to_change{
        time_step_id.slab_number(),
        db::get<::Tags::StepNumberWithinSlab>(*box) + delay_change_};

    db::mutate<
        evolution::dg::Tags::ChangeFixedLtsRatio::NumberOfExpectedMessages>(
        [&](const gsl::not_null<std::map<StepId, size_t>*> expected) {
          ++(*expected)[step_to_change];
        },
        box);

    if constexpr (Parallel::is_dg_element_collection_v<ParallelComponent>) {
      (void)cache, (void)array_index;
      ERROR("Reductions not implemented for DgElementCollection.");
    } else {
      if (synchronization_required) {
        const auto& component_proxy =
            Parallel::get_parallel_component<ParallelComponent>(cache);
        const auto& self_proxy = component_proxy[array_index];
        // As usual, Charm section stuff doesn't work very well, so we
        // can't send the result to just the section.
        Parallel::contribute_to_reduction<
            ChangeFixedLtsRatio_detail::StoreNewStep<Dim>>(
            ReductionData(section->id(), step_to_change, desired_step_size),
            self_proxy, component_proxy, make_not_null(&*section));
      } else {
        db::mutate<evolution::dg::Tags::ChangeFixedLtsRatio::NewStepSize>(
            [&](const gsl::not_null<std::map<StepId, std::vector<double>>*>
                    step_sizes) {
              (*step_sizes)[step_to_change].push_back(desired_step_size);
            },
            box);
      }
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

  void pup(PUP::er& p) override {
    p | step_choosers_;
    p | delay_change_;
    p | regions_;
  }

  std::unordered_map<std::type_index, StepperErrorTolerances> tolerances()
      const override {
    std::unordered_map<std::type_index, StepperErrorTolerances> tolerances{};
    for (const auto& step_chooser : step_choosers_) {
      collect_stepper_error_tolerances(&tolerances, *step_chooser);
    }
    return tolerances;
  }

 private:
  std::vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>
      step_choosers_;
  uint64_t delay_change_ = std::numeric_limits<uint64_t>::max();
  std::optional<std::vector<std::string>> regions_{};
};

template <size_t Dim, typename StepChoosersToUse>
// NOLINTNEXTLINE
PUP::able::PUP_ID ChangeFixedLtsRatio<Dim, StepChoosersToUse>::my_PUP_ID = 0;
}  // namespace dg::Events
