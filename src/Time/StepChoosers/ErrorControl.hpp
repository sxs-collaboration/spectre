// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "Options/String.hpp"
#include "Time/RequestsStepperErrorTolerances.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
struct NoSuchType;
/// \endcond

namespace StepChoosers {
namespace ErrorControl_detail {
template <typename StepChooserUse>
std::optional<double> goal_from_variable(
    const std::array<std::optional<StepperErrorEstimate>, 2>& errors,
    double min_factor, double max_factor, double safety_factor);
}  // namespace ErrorControl_detail

/*!
 * \brief Sets a goal based on time-stepper truncation error.
 *
 * \details The suggested step is calculated via a simple specialization of the
 * scheme suggested in \cite Hairer1993. We first compute the aggregated error
 * measure from the stepper error:
 *
 * \f[
 * E = \max_i(|E_i| / sc_i),
 * \f]
 *
 * where \f$E_i\f$ is the ODE error reported for each individual grid point,
 * reported by the time stepper, and \f$sc_i\f$ is the step control measure
 * determined by the tolerances:
 *
 * \f[
 * sc_i = Atol_i + \max(|y_i|,|y_i + E_i|) Rtol_i,
 * \f]
 *
 * and \f$y_i\f$ is the value of the function at the previous step at
 * grid point \f$i\f$.  (The estimate is more commonly done comparing
 * with the current step, but using the previous step avoids a memory
 * allocation in the TimeStepper and should not have a major effect on
 * the result.)
 *
 * When choosing a step size for LTS or when no record of previous
 * error is available, the step has size:
 *
 * \f[
 * h_{\text{new}} = h \cdot \min\left(F_{\text{max}},
 * \max\left(F_{\text{min}},
 * \frac{F_{\text{safety}}}{E^{1/(q + 1)}}\right)\right),
 * \f]
 *
 * where \f$h_{\text{new}}\f$ is the new suggested step size \f$h\f$ is the
 * previous step size, \f$F_{\text{max}}\f$ is the maximum factor by which we
 * allow the step to increase, \f$F_{\text{min}}\f$ is the minimum factor by
 * which we allow the step to decrease. \f$F_{\text{safety}}\f$ is the safety
 * factor on the computed error -- this forces the step size slightly lower
 * than we would naively compute so that the result of the step will likely be
 * within the target error. \f$q\f$ is the order of the stepper error
 * calculation. Intuitively, we should change the step less drastically for a
 * higher order stepper.
 *
 * When controlling slab size, after the first error calculation, the
 * error \f$E\f$ is recorded in the \ref DataBoxGroup "DataBox", and
 * subsequent error calculations use a simple PI scheme suggested in
 * \cite NumericalRecipes section 17.2.1:
 *
 * \f[
 * h_{\text{new}} = h \cdot \min\left(F_{\text{max}},
 * \max\left(F_{\text{min}},
 * F_{\text{safety}} E^{-0.7 / (q + 1)}
 * E_{\text{prev}}^{0.4 / (q + 1)}\right)\right),
 * \f]
 *
 * where \f$E_{\text{prev}}\f$ is the error computed in the previous
 * step.  This method is never used for choosing an LTS step because
 * the restriction of step size changes to factors of two was found to
 * interfere with the more gradual increase chosen by the PI
 * controller.
 */
template <typename StepChooserUse, typename System,
          typename = tmpl::conditional_t<
              tt::is_a_v<tmpl::list, typename System::variables_tag>,
              typename System::variables_tag,
              tmpl::list<typename System::variables_tag>>>
class ErrorControl;

template <typename StepChooserUse, typename System, typename... VariablesTags>
class ErrorControl<StepChooserUse, System, tmpl::list<VariablesTags...>>
    : public StepChooser<StepChooserUse>,
      public RequestsStepperErrorTolerances {
 public:
  /// \cond
  ErrorControl() = default;
  explicit ErrorControl(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ErrorControl);  // NOLINT
  /// \endcond

  struct AbsoluteTolerance {
    using type = double;
    static constexpr Options::String help{"Target absolute tolerance"};
    static type lower_bound() { return 0.0; }
  };

  struct RelativeTolerance {
    using type = double;
    static constexpr Options::String help{"Target relative tolerance"};
    static type lower_bound() { return 0.0; }
  };

  struct MaxFactor {
    using type = double;
    static constexpr Options::String help{
        "Maximum factor to increase the step by"};
    static type lower_bound() { return 1.0; }
  };

  struct MinFactor {
    using type = double;
    static constexpr Options::String help{
        "Minimum factor to increase the step by"};
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 1.0; }
  };

  struct SafetyFactor {
    using type = double;
    static constexpr Options::String help{
        "Extra factor to apply to step estimate; can be used to decrease step "
        "size to improve step acceptance rate."};
    static type lower_bound() { return 0.0; }
  };

  static constexpr Options::String help{
      "Sets a goal based on time-stepper truncation error."};
  using options = tmpl::list<AbsoluteTolerance, RelativeTolerance, MaxFactor,
                             MinFactor, SafetyFactor>;

  ErrorControl(const double absolute_tolerance, const double relative_tolerance,
               const double max_factor, const double min_factor,
               const double safety_factor)
      : absolute_tolerance_{absolute_tolerance},
        relative_tolerance_{relative_tolerance},
        max_factor_{max_factor},
        min_factor_{min_factor},
        safety_factor_{safety_factor} {}

  using argument_tags = tmpl::list<::Tags::StepperErrors<VariablesTags>...>;

  TimeStepRequest operator()(
      const typename ::Tags::StepperErrors<VariablesTags>::type&... errors,
      const double /*previous_step*/) const {
    const std::array goals{
        ErrorControl_detail::goal_from_variable<StepChooserUse>(
            errors, min_factor_, max_factor_, safety_factor_)...};
    std::optional<double> tightest_goal{};
    for (const auto& goal : goals) {
      if (goal.has_value() and (not tightest_goal.has_value() or
                                std::abs(*goal) < std::abs(*tightest_goal))) {
        tightest_goal = goal;
      }
    }
    return ::TimeStepRequest{.size_goal = tightest_goal};
  }

  bool uses_local_data() const override { return true; }
  bool can_be_delayed() const override { return true; }
  bool must_set_step_size() const override { return true; }

  std::unordered_map<std::type_index, StepperErrorTolerances> tolerances()
      const override {
    return {{typeid(VariablesTags),
             {.estimates = StepperErrorTolerances::Estimates::StepperOrder,
              .absolute = absolute_tolerance_,
              .relative = relative_tolerance_}}...};
  }

  void pup(PUP::er& p) override {  // NOLINT
    StepChooser<StepChooserUse>::pup(p);
    p | absolute_tolerance_;
    p | relative_tolerance_;
    p | min_factor_;
    p | max_factor_;
    p | safety_factor_;
  }

 private:
  double absolute_tolerance_ = std::numeric_limits<double>::signaling_NaN();
  double relative_tolerance_ = std::numeric_limits<double>::signaling_NaN();
  double max_factor_ = std::numeric_limits<double>::signaling_NaN();
  double min_factor_ = std::numeric_limits<double>::signaling_NaN();
  double safety_factor_ = std::numeric_limits<double>::signaling_NaN();
};
/// \cond
template <typename StepChooserUse, typename System, typename... VariablesTags>
PUP::able::PUP_ID
    ErrorControl<StepChooserUse, System,
                 tmpl::list<VariablesTags...>>::my_PUP_ID =  // NOLINT
    0;
/// \endcond
}  // namespace StepChoosers
