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
/// \cond
template <size_t DerivOrder>
struct OptionHolder;
/// \endcond

/// \ingroup ControlSystemGroup
/// All option tags related to the control system
namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup ControlSystemGroup
/// Options group for all control system options
struct ControlSystemGroup {
  static std::string name() { return "ControlSystems"; }
  static constexpr Options::String help = {
      "Options for all control systems used in a simulation."};
};

/// \ingroup OptionTagsGroup
/// \ingroup ControlSystemGroup
/// Option tag for each individual control system. The name of this option is
/// the name of the \p ControlSystem struct it is templated on. This way all
/// control systems will have a unique name.
template <typename ControlSystem>
struct ControlSystemInputs {
  static constexpr size_t deriv_order = ControlSystem::deriv_order;
  using type = control_system::OptionHolder<deriv_order>;
  static constexpr Options::String help{"Options for a control system."};
  static std::string name() { return ControlSystem::name(); }
  using group = ControlSystemGroup;
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
/// DataBox tag for all options of a single control system.
///
/// Only intended to be used during the initialization phase as a way of getting
/// options from multiple control systems into their corresponding components
/// DataBox.
template <typename ControlSystem>
struct ControlSystemInputs : db::SimpleTag {
  static constexpr size_t deriv_order = ControlSystem::deriv_order;
  using type = control_system::OptionHolder<deriv_order>;
  using option_tags =
      tmpl::list<OptionTags::ControlSystemInputs<ControlSystem>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& option) { return option; }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag for the averager
template <size_t DerivOrder>
struct Averager : db::SimpleTag {
  using type = ::Averager<DerivOrder>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag for the timescale tuner
struct TimescaleTuner : db::SimpleTag {
  using type = ::TimescaleTuner;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag for the controller
template <size_t DerivOrder>
struct Controller : db::SimpleTag {
  using type = ::Controller<DerivOrder>;
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

/// \ingroup ControlSystemGroup
/// Holds all options for a single control system
///
/// This struct collects all the options for a given control system during
/// option parsing. Then during initialization, the options can be retrieved via
/// their public member names and assigned to their corresponding DataBox tags.
template <size_t DerivOrder>
struct OptionHolder {
  struct Averager {
    using type = ::Averager<DerivOrder>;
    static constexpr Options::String help = {
        "Averages the derivatives of the control error and possibly the "
        "control error itself."};
  };

  struct Controller {
    using type = ::Controller<DerivOrder>;
    static constexpr Options::String help = {
        "Computes the control signal which will be used to reset the functions "
        "of time."};
  };

  struct TimescaleTuner {
    using type = ::TimescaleTuner;
    static constexpr Options::String help = {
        "Keeps track of the damping timescales for the control system upon "
        "which other timescales are based of off."};
  };

  using options = tmpl::list<Averager, Controller, TimescaleTuner>;
  static constexpr Options::String help = {"Options for a control system."};

  OptionHolder(::Averager<DerivOrder> input_averager,
               ::Controller<DerivOrder> input_controller,
               ::TimescaleTuner input_tuner)
      : averager(std::move(input_averager)),
        controller(std::move(input_controller)),
        tuner(std::move(input_tuner)) {}

  OptionHolder() = default;
  OptionHolder(const OptionHolder& /*rhs*/) = default;
  OptionHolder& operator=(const OptionHolder& /*rhs*/) = default;
  OptionHolder(OptionHolder&& /*rhs*/) = default;
  OptionHolder& operator=(OptionHolder&& /*rhs*/) = default;
  ~OptionHolder() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) {
    p | averager;
    p | controller;
    p | tuner;
  };

  // These members are specifically made pubic for easy access during
  // initialization
  ::Averager<DerivOrder> averager{};
  ::Controller<DerivOrder> controller{};
  ::TimescaleTuner tuner{};
};
}  // namespace control_system
