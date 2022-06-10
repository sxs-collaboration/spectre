// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/RootFinding/GslMultiRoot.hpp"

#include <ostream>

#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace RootFinder {
std::ostream& operator<<(std::ostream& os, const Method& method) {
  switch (method) {
    case Method::Hybrids:
      return os << "Hybrids";
    case Method::Hybrid:
      return os << "Hybrid";
    case Method::Newton:
      return os << "Newton";
    default:
      ERROR("Invalid method value specified.");
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const StoppingCondition& condition) {
  return call_with_dynamic_type<std::ostream&,
                                tmpl::list<StoppingConditions::Convergence,
                                           StoppingConditions::Residual>>(
      &condition, [&](const auto* c) -> decltype(auto) { return os << *c; });
}

namespace StoppingConditions {
std::ostream& operator<<(std::ostream& os, const Convergence& condition) {
  return os << "Convergence(abs=" << condition.absolute_tolerance
            << ", rel=" << condition.relative_tolerance << ")";
}
std::ostream& operator<<(std::ostream& os, const Residual& condition) {
  return os << "Residual(abs=" << condition.absolute_tolerance << ")";
}
}  // namespace StoppingConditions

namespace gsl_multiroot_detail {
void print_rootfinding_parameters(const Method method,
                                  const double maximum_absolute_tolerance,
                                  const StoppingCondition& condition) {
  Parallel::printf("\nAttempting a root find.\n");
  if (method == Method::Newton) {
    Parallel::printf(
        "Method: Newton. Modified to improve global convergence if analytic\n"
        "jacobian is provided.\n");
  } else if (method == Method::Hybrids) {
    Parallel::printf("Method: Scaled Hybrid.\n");
  } else if (method == Method::Hybrid) {
    Parallel::printf("Method: Unscaled Hybrid.\n");
  }
  Parallel::printf("Stopping condition: %s\n", condition);
  Parallel::printf(
      "A failed root find may still be \"forgiven\" (said to converge) if\n"
      "each component of f is below the maximum_absolute_tolerance provided.\n"
      "This value is zero by default, meaning that no failed root finds will\n"
      "be forgiven. Maximum absolute tolerance: %.17g\n",
      maximum_absolute_tolerance);
}
}  // namespace gsl_multiroot_detail
}  // namespace RootFinder
