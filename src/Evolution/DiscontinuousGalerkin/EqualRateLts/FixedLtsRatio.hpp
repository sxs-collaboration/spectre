// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/RequestsStepperErrorTolerances.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
namespace Tags {
struct FixedLtsRatio;
struct TimeStep;
}  // namespace Tags
namespace domain::Tags {
template <size_t VolumeDim>
struct Element;
}  // namespace domain::Tags
namespace evolution::dg::Tags {
template <size_t Dim>
struct EqualRateRegions;
}  // namespace evolution::dg::Tags
/// \endcond

namespace evolution::dg::StepChoosers {
/// Requests a slab size based on the desired step in regions with a
/// fixed slab fraction.
///
/// \note This StepChooser is not included in the
/// `standard_step_choosers` list.  Executables using the feature must
/// include it explicitly in the `factory_creation` struct and add the
/// `::Tags::FixedLtsRatio` tag to the element DataBox.
template <size_t Dim>
class FixedLtsRatio : public StepChooser<StepChooserUse::Slab>,
                      public RequestsStepperErrorTolerances {
 public:
  /// \cond
  FixedLtsRatio() = default;
  explicit FixedLtsRatio(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(FixedLtsRatio);  // NOLINT
  /// \endcond

  struct StepChoosers {
    using type =
        std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>;
    static constexpr Options::String help{"LTS step choosers to test"};
  };

  struct Regions {
    using type =
        Options::Auto<std::vector<std::string>, Options::AutoLabel::All>;
    static constexpr Options::String help{"Regions to adjust based on"};
  };

  static constexpr Options::String help{
      "Requests a slab size based on the desired step in regions with a fixed "
      "slab fraction."};
  using options = tmpl::list<StepChoosers, Regions>;

  explicit FixedLtsRatio(
      std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>
          step_choosers,
      std::optional<std::vector<std::string>> regions);

  using argument_tags = tmpl::list<::Tags::DataBox>;

  template <typename DbTags>
  TimeStepRequest operator()(const db::DataBox<DbTags>& box,
                             const double /*last_step*/) const {
    const auto& step_ratio = db::get<::Tags::FixedLtsRatio>(box);
    if (not step_ratio.has_value()) {
      return {};
    }

    if (regions_.has_value()) {
      const auto element_id = db::get<domain::Tags::Element<Dim>>(box).id();
      const auto& equal_rate_regions =
          db::get<Tags::EqualRateRegions<Dim>>(box);
      const auto& region_map = equal_rate_regions.regions();
      bool active = false;
      for (const auto& region_name : *regions_) {
        const auto region_it = region_map.find(region_name);
        if (region_it == region_map.end()) {
          ERROR_NO_TRACE("Unknown region name in FixedLtsRatio: "
                         << region_name
                         << ".  Known regions: " << keys_of(region_map));
        }
        if (equal_rate_regions.is_in_region(region_it->second, element_id)) {
          active = true;
          // Don't break here so we check the remaining names for validity.
        }
      }
      if (not active) {
        return {};
      }
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

  std::unordered_map<std::type_index, StepperErrorTolerances> tolerances()
      const override;

  void pup(PUP::er& p) override;

 private:
  std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>
      step_choosers_;
  std::optional<std::vector<std::string>> regions_;
};
}  // namespace evolution::dg::StepChoosers
