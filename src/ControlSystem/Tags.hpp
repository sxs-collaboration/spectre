// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/OptionTags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace control_system::Tags {
/// The measurement timescales associated with
/// domain::Tags::FunctionsOfTime.  Each function of time associated
/// with a control system has a corresponding set of timescales here,
/// represented as `PiecewisePolynomial<0>` with the same components
/// as the function itself.
struct MeasurementTimescales : db::SimpleTag {
  using type = std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;

  static constexpr bool pass_metavariables = true;

  template <typename Metavariables>
  using option_tags =
      tmpl::list<domain::OptionTags::DomainCreator<Metavariables::volume_dim>,
                 OptionTags::InitialTimeStep>;

  template <typename Metavariables>
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator,
      const double initial_time_step) noexcept {
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
}  // namespace control_system::Tags
