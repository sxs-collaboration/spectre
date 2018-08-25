// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/RootFinding/GslMultiRoot.hpp"

#include <ostream>

#include "ErrorHandling/Error.hpp"

namespace RootFinder {
std::ostream& operator<<(std::ostream& os,
                         const Verbosity& verbosity) noexcept {
  switch (verbosity) {
    case Verbosity::Silent:
      return os << "Silent";
    case Verbosity::Quiet:
      return os << "Quiet";
    case Verbosity::Verbose:
      return os << "Verbose";
    case Verbosity::Debug:
      return os << "Debug";
    default:
      ERROR("Invalid verbosity value specified.");
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const Method& method) noexcept {
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

std::ostream& operator<<(std::ostream& os,
                         const StoppingCondition& condition) noexcept {
  switch (condition) {
    case StoppingCondition::AbsoluteAndRelative:
      return os << "AbsoluteAndRelative";
    case StoppingCondition::Absolute:
      return os << "Absolute";
    default:
      ERROR("Invalid stopping condition specified.");
  }
  return os;
}

namespace gsl_multiroot_detail {
void print_rootfinding_parameters(const Method method,
                                  const double absolute_tolerance,
                                  const double relative_tolerance,
                                  const double maximum_absolute_tolerance,
                                  const StoppingCondition condition) noexcept {
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
  if (condition == StoppingCondition::Absolute) {
    Parallel::printf(
        "Stopping condition: Absolute. Convergence will be determined\n"
        "according to gsl_multiroot_test_residual.\n");
    Parallel::printf("Absolute tolerance: %.17g\n", absolute_tolerance);
  } else if (condition == StoppingCondition::AbsoluteAndRelative) {
    Parallel::printf(
        "Stopping condition: AbsoluteAndRelative. Convergence will be\n"
        "determined according to gsl_multiroot_test_delta.\n");
    Parallel::printf("Absolute tolerance: %.17g\n", absolute_tolerance);
    Parallel::printf("Relative tolerance: %.17g\n", relative_tolerance);
    if (relative_tolerance < 1.0e-13) {
      Parallel::printf(
          "Warning: using a relative tolerance below 1.0e-13 can result\n"
          "in an FPE coming from within gsl itself. Be wary of this if the\n"
          "root find FPEs without reporting any other error messages.");
    }
  }
  Parallel::printf(
      "A failed root find may still be \"forgiven\" (said to converge) if\n"
      "each component of f is below the maximum_absolute_tolerance provided.\n"
      "This value is zero by default, meaning that no failed root finds will\n"
      "be forgiven. Maximum absolute tolerance: %.17g\n",
      maximum_absolute_tolerance);
}
}  // namespace gsl_multiroot_detail
}  // namespace RootFinder
