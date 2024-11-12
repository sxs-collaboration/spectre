// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Time/ChangeSlabSize/Event.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/Tags/IsUsingTimeSteppingErrorControl.hpp"
#include "Time/Tags/StepperErrorTolerances.hpp"
#include "Time/Tags/StepperErrors.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
template <Triggers::WhenToCheck WhenToCheck>
struct EventsAndTriggers;
template <bool LocalTimeStepping>
struct IsUsingTimeSteppingErrorControlCompute;
struct StepChoosers;
template <typename EvolvedVariableTag, bool LocalTimeStepping>
struct StepperErrorTolerancesCompute;
}  // namespace Tags
/// \endcond

namespace StepChoosers {
namespace ErrorControl_detail {
struct IsAnErrorControl {};
template <typename EvolvedVariableTag>
struct ErrorControlBase : IsAnErrorControl {
 public:
  virtual StepperErrorTolerances tolerances() const = 0;

 protected:
  ErrorControlBase() = default;
  ~ErrorControlBase() = default;
};
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
 *
 * \note The template parameter `ErrorControlSelector` is used to disambiguate
 * in the input-file options between `ErrorControl` step choosers that are
 * based on different variables. This is needed if multiple systems are evolved
 * in the same executable. The name used for the input file includes
 * `ErrorControlSelector::name()` if it is provided.
 */
template <typename StepChooserUse, typename EvolvedVariableTag,
          typename ErrorControlSelector = NoSuchType>
class ErrorControl
    : public StepChooser<StepChooserUse>,
      public ErrorControl_detail::ErrorControlBase<EvolvedVariableTag> {
 public:
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

  using simple_tags = tmpl::list<::Tags::StepperErrors<EvolvedVariableTag>>;

  using compute_tags = tmpl::list<
      Tags::IsUsingTimeSteppingErrorControlCompute<
          std::is_same_v<StepChooserUse, ::StepChooserUse::LtsStep>>,
      Tags::StepperErrorTolerancesCompute<
          EvolvedVariableTag,
          std::is_same_v<StepChooserUse, ::StepChooserUse::LtsStep>>>;

  using argument_tags = tmpl::list<::Tags::StepperErrors<EvolvedVariableTag>>;

  std::pair<TimeStepRequest, bool> operator()(
      const typename ::Tags::StepperErrors<EvolvedVariableTag>::type& errors,
      const double previous_step) const {
    // Do not request that the step size be changed if there isn't a new error
    // estimate
    if (not errors[1].has_value()) {
      return {{}, true};
    }
    double new_step;
    if (std::is_same_v<StepChooserUse, ::StepChooserUse::LtsStep> or
        not errors[0].has_value() or errors[0]->order != errors[1]->order) {
      new_step = errors[1]->step_size.value() *
                 std::clamp(safety_factor_ *
                                pow(1.0 / std::max(errors[1]->error, 1e-14),
                                    1.0 / (errors[1]->order + 1)),
                            min_factor_, max_factor_);
    } else {
      // From simple advice from Numerical Recipes 17.2.1 regarding a heuristic
      // for PI step control.
      const double alpha_factor = 0.7 / (errors[1]->order + 1);
      const double beta_factor = 0.4 / (errors[0]->order + 1);
      new_step =
          errors[1]->step_size.value() *
          std::clamp(
              safety_factor_ *
                  pow(1.0 / std::max(errors[1]->error, 1e-14), alpha_factor) *
                  pow(std::max(errors[0]->error, 1e-14), beta_factor),
              min_factor_, max_factor_);
    }
    // If the error is out-of-date we can still reject a step, but it
    // won't improve the reported error, so make sure to only do it
    // if we actually request a smaller step.
    return {::TimeStepRequest{.size_goal = new_step},
            abs(new_step) >= abs(previous_step) or errors[1]->error <= 1.0};
  }

  bool uses_local_data() const override { return true; }
  bool can_be_delayed() const override { return true; }

  StepperErrorTolerances tolerances() const override {
    return {.absolute = absolute_tolerance_, .relative = relative_tolerance_};
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
  using return_type = type;
  using argument_tags = tmpl::conditional_t<
      LocalTimeStepping, tmpl::list<::Tags::StepChoosers>,
      tmpl::list<::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>>;

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

/// \ingroup TimeGroup
/// \brief A tag that contains the error tolerances if the
/// `ErrorControl` step chooser is one of the option-created `Event`s.
template <typename EvolvedVariableTag, bool LocalTimeStepping>
struct StepperErrorTolerancesCompute
    : db::ComputeTag,
      StepperErrorTolerances<EvolvedVariableTag> {
  using base = StepperErrorTolerances<EvolvedVariableTag>;
  using return_type = typename base::type;
  using argument_tags = tmpl::conditional_t<
      LocalTimeStepping, tmpl::list<::Tags::StepChoosers>,
      tmpl::list<::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>>;

  // local time stepping
  static void function(
      const gsl::not_null<std::optional<::StepperErrorTolerances>*> tolerances,
      const std::vector<
          std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>&
          step_choosers) {
    tolerances->reset();
    for (const auto& step_chooser : step_choosers) {
      set_tolerances_if_error_control(tolerances, *step_chooser);
    }
  }

  // global time stepping
  static void function(
      const gsl::not_null<std::optional<::StepperErrorTolerances>*> tolerances,
      const ::EventsAndTriggers& events_and_triggers) {
    tolerances->reset();
    // In principle the slab size could be changed based on a dense
    // trigger, but it's not clear that there is ever a good reason to
    // do so, and it wouldn't make sense to use error control in that
    // context in any case.
    events_and_triggers.for_each_event([&](const auto& event) {
      if (const auto* const change_slab_size =
              dynamic_cast<const ::Events::ChangeSlabSize*>(&event)) {
        change_slab_size->for_each_step_chooser(
            [&](const StepChooser<StepChooserUse::Slab>& step_chooser) {
              set_tolerances_if_error_control(tolerances, step_chooser);
            });
      }
    });
  }

 private:
  template <typename StepChooserUse>
  static void set_tolerances_if_error_control(
      const gsl::not_null<std::optional<::StepperErrorTolerances>*> tolerances,
      const StepChooser<StepChooserUse>& step_chooser) {
    if (const auto* const error_control =
            dynamic_cast<const ::StepChoosers::ErrorControl_detail::
                             ErrorControlBase<EvolvedVariableTag>*>(
                &step_chooser);
        error_control != nullptr) {
      const auto this_tolerances = error_control->tolerances();
      if (tolerances->has_value() and tolerances->value() != this_tolerances) {
        ERROR_NO_TRACE("All ErrorControl events for "
                       << db::tag_name<EvolvedVariableTag>()
                       << " must use the same tolerances.");
      }
      tolerances->emplace(this_tolerances);
    }
  }
};
}  // namespace Tags
