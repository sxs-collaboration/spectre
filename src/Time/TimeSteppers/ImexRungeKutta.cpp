// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/ImexRungeKutta.hpp"

#include "Time/History.hpp"
#include "Time/Time.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace TimeSteppers {

namespace {
template <typename T>
void apply_explicit_coefficients(const gsl::not_null<T*> u,
                                 const ConstUntypedHistory<T>& implicit_history,
                                 const std::vector<double>& coefficients,
                                 const size_t number_of_coefficients_to_apply,
                                 const double time_step) {
  if (number_of_coefficients_to_apply > 0) {
    if (coefficients[0] != 0.0) {
      *u += coefficients[0] * time_step * implicit_history.back().derivative;
    }
    for (size_t i = 1; i < number_of_coefficients_to_apply; ++i) {
      if (coefficients[i] != 0.0) {
        *u += coefficients[i] * time_step *
              implicit_history.substeps()[i - 1].derivative;
      }
    }
  }
}
}  // namespace

template <typename T>
void ImexRungeKutta::add_inhomogeneous_implicit_terms_impl(
    const gsl::not_null<T*> u, const ConstUntypedHistory<T>& implicit_history,
    const TimeDelta& time_step) const {
  ASSERT(implicit_history.integration_order() == order(),
         "Fixed-order stepper cannot run at order "
             << implicit_history.integration_order());

  auto substep =
      implicit_history.at_step_start() ? 0 : implicit_history.substeps().size();

  ASSERT(number_of_substeps() == number_of_substeps_for_error(),
         "The current interface does not provide enough information to "
         "determine whether error estimation is active in IMEX functions.");
  if (substep == number_of_substeps() - 1) {
    const auto& coefficients = butcher_tableau().result_coefficients;
    apply_explicit_coefficients(u, implicit_history, coefficients,
                                coefficients.size(), time_step.value());
  } else {
    ASSERT(substep < implicit_butcher_tableau().substep_coefficients.size(),
           "Tableau too short: " << substep << "/"
           << implicit_butcher_tableau().substep_coefficients.size());
    const auto& coefficients =
        implicit_butcher_tableau().substep_coefficients[substep];
    apply_explicit_coefficients(u, implicit_history, coefficients,
                                std::min(substep + 1, coefficients.size()),
                                time_step.value());
  }
}

template <typename T>
double ImexRungeKutta::implicit_weight_impl(
    const ConstUntypedHistory<T>& implicit_history,
    const TimeDelta& time_step) const {
  const auto substep =
      implicit_history.at_step_start() ? 0 : implicit_history.substeps().size();
  const auto& coefficients = implicit_butcher_tableau().substep_coefficients;
  if (coefficients.size() > substep and
      coefficients[substep].size() == substep + 2) {
    return time_step.value() * coefficients[substep][substep + 1];
  } else {
    return 0.0;
  }
}

IMEX_TIME_STEPPER_DEFINE_OVERLOADS(ImexRungeKutta)
}  // namespace TimeSteppers
