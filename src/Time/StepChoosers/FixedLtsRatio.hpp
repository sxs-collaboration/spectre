// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Options/String.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
namespace Tags {
struct FixedLtsRatio;
struct TimeStep;
}  // namespace Tags
/// \endcond

namespace StepChoosers {
/// Requests a slab size based on the desired step in regions with a
/// fixed slab fraction.
///
/// \note This StepChooser is not included in the
/// `standard_step_choosers` list.  Executables using the feature must
/// include it explicitly in the `factory_creation` struct and add the
/// `::Tags::FixedLtsRatio` tag to the element DataBox.
class FixedLtsRatio
    : public SPECTRE_CHARM_DERIVED(
          FixedLtsRatio, SINGLE_ARG(StepChooser<StepChooserUse::Slab>)) {
 public:
  /// \cond
  FixedLtsRatio() = default;
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(FixedLtsRatio);  // NOLINT
  /// \endcond

  struct StepChoosers {
    using type =
        std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>;
    static constexpr Options::String help{"LTS step choosers to test"};
  };

  static constexpr Options::String help{
      "Requests a slab size based on the desired step in regions with a fixed "
      "slab fraction."};
  using options = tmpl::list<StepChoosers>;

  explicit FixedLtsRatio(
      std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>
          step_choosers);

  using argument_tags = tmpl::list<::Tags::DataBox>;

  template <typename DbTags>
  TimeStepRequest operator()(const db::DataBox<DbTags>& box,
                             const double /*last_step*/) const {
    const auto& step_ratio = db::get<::Tags::FixedLtsRatio>(box);
    if (not step_ratio.has_value()) {
      return {};
    }

    const auto& current_step = db::get<::Tags::TimeStep>(box);
    const evolution_less<double> less{current_step.is_positive()};

    std::optional<double> size_goal{};
    std::optional<double> size{};
    for (const auto& step_chooser : step_choosers_) {
      const auto step_request =
          step_chooser->desired_step(current_step.value(), box);

      if (step_request.size_goal.has_value()) {
        if (size_goal.has_value()) {
          *size_goal = std::min(*size_goal, *step_request.size_goal, less);
        } else {
          size_goal = step_request.size_goal;
        }
      }
      if (step_request.size.has_value()) {
        if (size.has_value()) {
          *size = std::min(*size, *step_request.size, less);
        } else {
          size = step_request.size;
        }
      }

      // As of writing (Oct. 2024), no StepChooserUse::LtsStep chooser
      // sets these.
      ASSERT(not(step_request.end.has_value() or
                 step_request.size_hard_limit.has_value() or
                 step_request.end_hard_limit.has_value()),
             "Unhandled field set by StepChooser.  Please file a bug "
             "containing the options passed to FixedLtsRatio.");
    }

    if (size_goal.has_value()) {
      *size_goal *= *step_ratio;
    }
    if (size.has_value()) {
      *size *= *step_ratio;
      if (size_goal.has_value() and less(*size_goal, *size)) {
        // Not allowed to request a goal and a bigger step.
        size.reset();
      }
    }

    return {.size_goal = size_goal, .size = size};
  }

  bool uses_local_data() const override;
  bool can_be_delayed() const override;

  template <typename F>
  void for_each_step_chooser(F&& f) const {
    for (const auto& step_chooser : step_choosers_) {
      f(*step_chooser);
    }
  }

  void pup(PUP::er& p) override;

 private:
  std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>
      step_choosers_;
};
}  // namespace StepChoosers
