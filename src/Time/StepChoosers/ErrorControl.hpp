// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "Time/Actions/ChangeSlabSize.hpp"
#include "Time/StepChoosers/StepChooser.hpp"  // IWYU pragma: keep
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsIterable.hpp"

/// \cond
namespace Tags {
template <bool LocalTimeStepping>
struct IsUsingTimeSteppingErrorControlCompute;
}  // namespace Tags
/// \endcond

namespace StepChoosers {
namespace ErrorControl_detail {
struct IsAnErrorControl {};
}  // namespace ErrorControl_detail

/*!
 * \brief Suggests a step size based on a target absolute and/or relative error
 * measure.
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
 * and \f$y_i\f$ is the value of the function at the current step at grid point
 * \f$i\f$.
 *
 * When no previous record of previous error is available, the step has size:
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
 * After the first error calculation, the error \f$E\f$ is recorded in the \ref
 * DataBoxGroup "DataBox" using tag
 * `StepChoosers::Tags::PreviousStepError<EvolvedVariablesTag>`, and subsequent
 * error calculations use a simple PI scheme suggested in \cite NumericalRecipes
 * section 17.2.1:
 *
 * \f[
 * h_{\text{new}} = h \cdot \min\left(F_{\text{max}},
 * \max\left(F_{\text{min}},
 * F_{\text{safety}} E^{-0.7 / (q + 1)}
 * E_{\text{prev}}^{0.4 / (q + 1)}\right)\right),
 * \f]
 *
 * where \f$E_{\text{prev}}\f$ is the error computed in the previous step.
 *
 * \note The template parameter `ErrorControlSelector` is used to disambiguate
 * in the input-file options between `ErrorControl` step choosers that are
 * based on different variables. This is needed if multiple systems are evolved
 * in the same executable. The name used for the input file includes
 * `ErrorControlSelector::name()` if it is provided.
 */
template <typename StepChooserUse, typename EvolvedVariableTag,
          typename ErrorControlSelector = NoSuchType>
class ErrorControl : public StepChooser<StepChooserUse>,
                     public ErrorControl_detail::IsAnErrorControl {
 public:
  using evolved_variable_type = typename EvolvedVariableTag::type;
  using error_variable_type =
      typename ::Tags::StepperError<EvolvedVariableTag>::type;

  /// \cond
  ErrorControl() = default;
  explicit ErrorControl(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ErrorControl);  // NOLINT
  /// \endcond

  static std::string name() {
    if constexpr (std::is_same_v<ErrorControlSelector, NoSuchType>) {
      return "ErrorControl";
    } else {
      return "ErrorControl(" + ErrorControlSelector::name() + ")";
    }
  }

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
      "Chooses a step based on a target relative and absolute error tolerance"};
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

  using simple_tags =
      tmpl::list<::Tags::StepperError<EvolvedVariableTag>,
                 ::Tags::PreviousStepperError<EvolvedVariableTag>,
                 ::Tags::StepperErrorUpdated>;

  using compute_tags = tmpl::list<Tags::IsUsingTimeSteppingErrorControlCompute<
      std::is_same_v<StepChooserUse, ::StepChooserUse::LtsStep>>>;

  using argument_tags =
      tmpl::list<::Tags::HistoryEvolvedVariables<EvolvedVariableTag>,
                 ::Tags::StepperError<EvolvedVariableTag>,
                 ::Tags::PreviousStepperError<EvolvedVariableTag>,
                 ::Tags::StepperErrorUpdated, ::Tags::TimeStepper<>>;

  std::pair<double, bool> operator()(
      const TimeSteppers::History<evolved_variable_type>& history,
      const error_variable_type& error,
      const error_variable_type& previous_error,
      const bool& stepper_error_updated, const TimeStepper& stepper,
      const double previous_step) const {
    // request that the step size not be changed if there isn't a new error
    // estimate
    if (not stepper_error_updated) {
      return std::make_pair(std::numeric_limits<double>::infinity(), true);
    }
    const double l_inf_error = error_calc_impl(history.latest_value(), error);
    double new_step;
    if (previous_error.number_of_grid_points() !=
        history.latest_value().number_of_grid_points()) {
      new_step = previous_step *
                 std::clamp(safety_factor_ *
                                pow(1.0 / std::max(l_inf_error, 1e-14),
                                    1.0 / (stepper.error_estimate_order() + 1)),
                            min_factor_, max_factor_);
    } else {
      const double previous_l_inf_error =
          error_calc_impl(history.latest_value(), previous_error);
      // From simple advice from Numerical Recipes 17.2.1 regarding a heuristic
      // for PI step control.
      const double alpha_factor = 0.7 / (stepper.error_estimate_order() + 1);
      const double beta_factor = 0.4 / (stepper.error_estimate_order() + 1);
      new_step =
          previous_step *
          std::clamp(
              safety_factor_ *
                  pow(1.0 / std::max(l_inf_error, 1e-14), alpha_factor) *
                  pow(std::max(previous_l_inf_error, 1e-14), beta_factor),
              min_factor_, max_factor_);
    }
    return std::make_pair(new_step, l_inf_error <= 1.0);
  }

  bool uses_local_data() const override { return true; }

  void pup(PUP::er& p) override {  // NOLINT
    p | absolute_tolerance_;
    p | relative_tolerance_;
    p | min_factor_;
    p | max_factor_;
    p | safety_factor_;
  }

 private:
  template <typename EvolvedType, typename ErrorType>
  double error_calc_impl(const EvolvedType& values,
                         const ErrorType& errors) const {
    if constexpr (std::is_fundamental_v<std::remove_cv_t<EvolvedType>> or
                  tt::is_complex_of_fundamental_v<
                      std::remove_cv_t<EvolvedType>>) {
      return std::abs(errors) /
             (absolute_tolerance_ +
              relative_tolerance_ *
                  std::max(abs(values), abs(values + errors)));
    } else if constexpr (tt::is_iterable_v<std::remove_cv_t<EvolvedType>>) {
      double result = 0.0;
      double recursive_call_result;
      for (auto val_it = values.begin(), err_it = errors.begin();
           val_it != values.end(); ++val_it, ++err_it) {
        recursive_call_result = error_calc_impl(*val_it, *err_it);
        if (recursive_call_result > result) {
          result = recursive_call_result;
        }
      }
      return result;
    } else if constexpr (tt::is_a_v<Variables, std::remove_cv_t<EvolvedType>>) {
      double result = 0.0;
      double recursive_call_result;
      tmpl::for_each<typename EvolvedType::tags_list>(
          [this, &errors, &values, &recursive_call_result,
           &result](auto tag_v) {
            // clang erroneously warns that `this` is not used despite requiring
            // it in the capture...
            (void)this;
            using tag = typename decltype(tag_v)::type;
            recursive_call_result =
                error_calc_impl(get<tag>(values), get<tag>(errors));
            if (recursive_call_result > result) {
              result = recursive_call_result;
            }
          });
      return result;
    } else if constexpr (is_any_spin_weighted_v<
                             std::remove_cv_t<EvolvedType>>) {
      return error_calc_impl(values.data(), errors.data());
    }
  }

  double absolute_tolerance_ = std::numeric_limits<double>::signaling_NaN();
  double relative_tolerance_ = std::numeric_limits<double>::signaling_NaN();
  double max_factor_ = std::numeric_limits<double>::signaling_NaN();
  double min_factor_ = std::numeric_limits<double>::signaling_NaN();
  double safety_factor_ = std::numeric_limits<double>::signaling_NaN();
};
/// \cond
template <typename StepChooserUse, typename EvolvedVariableTag,
          typename ErrorControlSelector>
PUP::able::PUP_ID ErrorControl<StepChooserUse, EvolvedVariableTag,
                               ErrorControlSelector>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace StepChoosers

namespace Tags {
/// \ingroup TimeGroup
/// \brief A tag that is true if the `ErrorControl` step chooser is one of the
/// option-created `Event`s.
template <bool LocalTimeStepping>
struct IsUsingTimeSteppingErrorControlCompute
    : db::ComputeTag,
    IsUsingTimeSteppingErrorControl {
  using base = IsUsingTimeSteppingErrorControl;
  using argument_tags =
      tmpl::conditional_t<LocalTimeStepping, tmpl::list<::Tags::StepChoosers>,
                          tmpl::list<::Tags::EventsAndTriggers>>;

  // local time stepping
  static void function(
      const gsl::not_null<bool*> is_using_error_control,
      const std::vector<
          std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>&
          step_choosers) {
    *is_using_error_control = false;
    for (const auto& step_chooser : step_choosers) {
      if (dynamic_cast<
              const ::StepChoosers::ErrorControl_detail::IsAnErrorControl*>(
              &*step_chooser) != nullptr) {
        *is_using_error_control = true;
        return;
      }
    }
  }

  // global time stepping
  static void function(const gsl::not_null<bool*> is_using_error_control,
                       const ::EventsAndTriggers& events_and_triggers) {
    // In principle the slab size could be changed based on a dense
    // trigger, but it's not clear that there is ever a good reason to
    // do so, and it wouldn't make sense to use error control in that
    // context in any case.
    *is_using_error_control = false;
    events_and_triggers.for_each_event([&](const auto& event) {
      if (*is_using_error_control) {
        return;
      }
      if (const auto* const change_slab_size =
              dynamic_cast<const ::Events::ChangeSlabSize*>(&event)) {
        change_slab_size->for_each_step_chooser(
            [&](const StepChooser<StepChooserUse::Slab>& step_chooser) {
              if (*is_using_error_control) {
                return;
              }
              if (dynamic_cast<const ::StepChoosers::ErrorControl_detail::
                                   IsAnErrorControl*>(&step_chooser) !=
                  nullptr) {
                *is_using_error_control = true;
              }
            });
      }
    });
  }
};
}  // namespace Tags
