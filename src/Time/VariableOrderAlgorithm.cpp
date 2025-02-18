// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/VariableOrderAlgorithm.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <pup.h>

#include "Time/StepperErrorEstimate.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

VariableOrderAlgorithm::VariableOrderAlgorithm() = default;

VariableOrderAlgorithm::VariableOrderAlgorithm(const size_t goal_order)
    : goal_order_(goal_order) {}

VariableOrderAlgorithm::VariableOrderAlgorithm(const double order_falloff)
    : order_falloff_(order_falloff) {}

StepperErrorTolerances::Estimates VariableOrderAlgorithm::required_estimates()
    const {
  return order_falloff_.has_value()
             ? StepperErrorTolerances::Estimates::AllOrders
             : StepperErrorTolerances::Estimates::None;
}

void VariableOrderAlgorithm::pup(PUP::er& p) {
  p | goal_order_;
  p | order_falloff_;
}

size_t VariableOrderAlgorithm::choose_order_goal(
    const size_t current_order) const {
  if (current_order == goal_order_.value()) {
    return current_order;
  } else if (current_order < goal_order_.value()) {
    return current_order + 1;
  } else {
    return current_order - 1;
  }
}

namespace {
template <size_t NVars>
double largest_error(
    const std::array<const std::optional<StepperErrorEstimate>*, NVars>& errors,
    const size_t order) {
  double order_error = -std::numeric_limits<double>::infinity();
  for (const auto* error : errors) {
    if (error->has_value()) {
      order_error = std::max(order_error,
                             gsl::at(error->value().errors, order - 1).value());
    }
  }

  if (order_error == -std::numeric_limits<double>::infinity()) {
    ERROR_NO_TRACE(
        "OrderFalloff only implemented with error-based adaptive time "
        "stepping.");
  }

  return order_error;
}
}  // namespace

template <size_t NVars>
size_t VariableOrderAlgorithm::choose_order_falloff(
    const size_t current_order,
    const std::array<const std::optional<StepperErrorEstimate>*, NVars>& errors)
    const {
  double prev_error = largest_error(errors, 1);
  double best_convergence = 1.0;
  for (size_t order = 2; order < current_order; ++order) {
    const double error = largest_error(errors, order);
    const double convergence = error / prev_error;
    if (convergence > std::pow(best_convergence, order_falloff_.value())) {
      return current_order - 1;
    }
    prev_error = error;
    best_convergence = std::min(best_convergence, convergence);
  }

  const double current_error = largest_error(errors, current_order);

  const double current_convergence = current_error / prev_error;
  if (current_convergence >
      std::pow(best_convergence, order_falloff_.value())) {
    return current_order;
  }

  return current_order + 1;
}

bool operator==(const VariableOrderAlgorithm& a,
                const VariableOrderAlgorithm& b) {
  return a.goal_order_ == b.goal_order_ and
         a.order_falloff_ == b.order_falloff_;
}

#define NVARS(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                       \
  template size_t VariableOrderAlgorithm::choose_order_falloff(    \
      size_t current_order,                                        \
      const std::array<const std::optional<StepperErrorEstimate>*, \
                       NVARS(data)>& errors) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2))

#undef INSTATNTIATE
#undef NVARS
