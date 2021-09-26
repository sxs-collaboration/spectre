// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "ControlSystem/Averager.hpp"
#include "ControlSystem/Controller.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/OptionTags.hpp"
#include "Options/Options.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace control_system {

/// \ingroup ControlSystemGroup
/// All option tags related to the control system
namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup ControlSystemGroup
/// Options group for all control system options
struct ControlSystemGroup {
  static std::string name() { return "ControlSystem"; }
  static constexpr Options::String help = {"All options for a control system."};
};

/// \ingroup OptionTagsGroup
/// \ingroup ControlSystemGroup
/// Options for the averager
template <size_t DerivOrder>
struct Averager {
  using type = ::Averager<DerivOrder>;
  static constexpr Options::String help = {"Options for the averager."};
  using group = OptionTags::ControlSystemGroup;
};

/// \ingroup OptionTagsGroup
/// \ingroup ControlSystemGroup
/// Options for the controller
template <size_t DerivOrder>
struct Controller {
  using type = ::Controller<DerivOrder>;
  static constexpr Options::String help = {"Options for the controller."};
  using group = OptionTags::ControlSystemGroup;
};

/// \ingroup OptionTagsGroup
/// \ingroup ControlSystemGroup
/// Options for the timescale tuner
struct TimescaleTuner {
  using type = ::TimescaleTuner;
  static constexpr Options::String help = {"Options for the timescale tuner."};
  using group = OptionTags::ControlSystemGroup;
};
}  // namespace OptionTags

/// \ingroup ControlSystemGroup
/// All DataBox tags related to the control system
namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag for the name of a control system
struct ControlSystemName : db::SimpleTag {
  using type = std::string;
};
/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag for the averager
template <size_t DerivOrder>
struct Averager : db::SimpleTag {
  using type = ::Averager<DerivOrder>;

  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::Averager<DerivOrder>>;

  static auto create_from_options(
      const ::Averager<DerivOrder>& averager) {
    return averager;
  }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag for the timescale tuner
struct TimescaleTuner : db::SimpleTag {
  using type = ::TimescaleTuner;

  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::TimescaleTuner>;

  static auto create_from_options(
      const ::TimescaleTuner& timescale_tuner) {
    return timescale_tuner;
  }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag for the controller
template <size_t DerivOrder>
struct Controller : db::SimpleTag {
  using type = ::Controller<DerivOrder>;

  static constexpr bool pass_metavariables = false;

  using option_tags = tmpl::list<OptionTags::Controller<DerivOrder>>;

  static auto create_from_options(
      const ::Controller<DerivOrder>& controller) {
    return controller;
  }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// \brief The measurement timescales associated with
/// domain::Tags::FunctionsOfTime.
///
/// Each function of time associated with a control system has a corresponding
/// set of timescales here, represented as `PiecewisePolynomial<0>` with the
/// same components as the function itself.
struct MeasurementTimescales : db::SimpleTag {
  using type = std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;

  static constexpr bool pass_metavariables = true;

  template <typename Metavariables>
  using option_tags =
      tmpl::list<domain::OptionTags::DomainCreator<Metavariables::volume_dim>,
                 ::OptionTags::InitialTimeStep>;

  template <typename Metavariables>
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator,
      const double initial_time_step) {
    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        timescales;
    for (const auto& function_of_time : domain_creator->functions_of_time()) {
      if (function_of_time.second->time_bounds()[1] ==
          std::numeric_limits<double>::infinity()) {
        // This function of time is not controlled by a control
        // system.  It is an analytic function or similar.
        continue;
      }
      const double function_initial_time =
          function_of_time.second->time_bounds()[0];
      const DataVector used_for_size =
          function_of_time.second->func(function_initial_time)[0];

      // This check is intentionally inside the loop over the
      // functions of time so that it will not trigger for domains
      // without control systems.
      if (initial_time_step <= 0.0) {
        ERROR(
            "Control systems can only be used in forward-in-time evolutions.");
      }

      auto initial_timescale =
          make_with_value<DataVector>(used_for_size, initial_time_step);
      timescales.emplace(
          function_of_time.first,
          std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
              function_initial_time, std::array{std::move(initial_timescale)},
              std::numeric_limits<double>::infinity()));
    }
    return timescales;
  }
};
}  // namespace Tags
}  // namespace control_system
