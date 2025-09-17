// Distributed under the MIT License.
// See LICENSE.txt for details.

// Tests for the VariableOrderAlgorithm class are performed in
// Test_ChangeTimeStepperOrder.cpp.  The class is split into this file
// to avoid include loops with the associated tags.

#pragma once

#include <array>
#include <cstddef>
#include <optional>

#include "Options/String.hpp"
#include "Time/History.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Options {
template <typename... AlternativeLists>
struct Alternatives;
}  // namespace Options
/// \endcond

/*!
 * \ingroup TimeGroup
 * \brief Class encapsulating the time-stepper order changing algorithms
 *
 * \details Supports two modes: driving the order to a specified constant, and
 * measuring the convergence of the error estimates.
 *
 * When driving to a constant, the order is changed by one towards the
 * goal if it is not at the goal.
 *
 * When measuring convergence, let the relative error estimate from
 * the time stepper for order-$k$ integration be $e_k$, and $\lambda
 * \le 1$ be the falloff this class was constructed with.  Then, we
 * decrease the order by one if, for any $k \ge 2$ and less than the
 * current integration order,
 *
 * \f{equation}
 *   \frac{e_k}{e_{k-1}} >
 *   \left(\min_{2 \le j < k} \frac{e_j}{e_{j-1}}\right)^\lambda,
 * \f}
 *
 * with the right-hand side taken to be $1$ for $k = 2$.  If the order
 * is not decreased, it is kept the same if the above condition holds
 * for $k$ equal to the current integration order, and is increased by
 * one otherwise.  If the current integration order is $1$ (not
 * possible for predictor-corrector methods), it is always increased.
 * This algorithm will almost never prefer an integration order less
 * than three.
 *
 * \note This is currently all implemented in one class for simplicity
 * with dealing with templates.  If we add more algorithms splitting
 * into a base class with implementations would be appropriate.
 */
class VariableOrderAlgorithm {
 public:
  struct GoalOrder {
    using type = size_t;
    static constexpr Options::String help = "Order to drive the integrator to.";
  };

  struct OrderFalloff {
    using type = double;
    static constexpr Options::String help =
        "Threshold for changing time-stepper order, as a logarithmic "
        "fraction of the best order-to-order improvement.";
  };

  using options = tmpl::list<
      Options::Alternatives<tmpl::list<GoalOrder>, tmpl::list<OrderFalloff>>>;
  static constexpr Options::String help =
      "Algorithm for choosing the time-stepper order in a variable-order "
      "evolution.";

  VariableOrderAlgorithm();
  explicit VariableOrderAlgorithm(size_t goal_order);
  explicit VariableOrderAlgorithm(double order_falloff);

  StepperErrorTolerances::Estimates required_estimates() const;

  template <typename... VariablesTags>
  size_t choose_order(
      const TimeSteppers::History<typename VariablesTags::type>&... histories,
      const typename tmpl::has_type<
          VariablesTags,
          std::array<std::optional<StepperErrorEstimate>, 2>>::type&... errors)
      const {
    const auto history_order =
        get_first_argument(histories...).integration_order();
    ASSERT((... and (history_order == histories.integration_order())),
           "Multiple histories with different integration orders.");

    if (goal_order_.has_value()) {
      ASSERT(not order_falloff_.has_value(),
             "Internal error: should not have multiple algorithms active");
      return choose_order_goal(history_order);
    } else {
      ASSERT(order_falloff_.has_value(),
             "VariableOrderAlgorithm not initialized");
      return choose_order_falloff(history_order, std::array{&errors[1]...});
    }
  }

  void pup(PUP::er& p);

 private:
  size_t choose_order_goal(size_t current_order) const;

  template <size_t NVars>
  size_t choose_order_falloff(
      size_t current_order,
      const std::array<const std::optional<StepperErrorEstimate>*, NVars>&
          errors) const;

  friend bool operator==(const VariableOrderAlgorithm& a,
                         const VariableOrderAlgorithm& b);

  std::optional<size_t> goal_order_{};
  std::optional<double> order_falloff_{};
};
