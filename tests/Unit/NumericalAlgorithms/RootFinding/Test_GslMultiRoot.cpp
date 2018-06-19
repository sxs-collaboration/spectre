// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <string>
#include <vector>

#include "ErrorHandling/Exceptions.hpp"
#include "NumericalAlgorithms/RootFinding/GslMultiRoot.hpp"
#include "Utilities/GetOutput.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

// Rosenbrock system of equations:
// f_1 (x0, x1) = a (1 - x0)
// f_2 (x0, x1) = b (x1 - x0**2)
class Rosenbrock {
 public:
  Rosenbrock(const double& a, const double& b) : a_(a), b_(b) {}
  std::array<double, 2> operator()(const std::array<double, 2>& x) const
      noexcept {
    const double y0 = a_ * (1.0 - x[0]);
    const double y1 = b_ * (x[1] - x[0] * x[0]);
    return std::array<double, 2>{{y0, y1}};
  }
  std::array<std::array<double, 2>, 2> jacobian(
      const std::array<double, 2>& x) const noexcept {
    return std::array<std::array<double, 2>, 2>{
        {{{-a_, 0.0}}, {{-2.0 * b_ * x[0], b_}}}};
  }

 private:
  double a_, b_;
};

class RosenbrockNoJac {
 public:
  RosenbrockNoJac(const double& a, const double& b) : a_(a), b_(b) {}
  std::array<double, 2> operator()(const std::array<double, 2>& x) const
      noexcept {
    const double y0 = a_ * (1.0 - x[0]);
    const double y1 = b_ * (x[1] - x[0] * x[0]);
    return std::array<double, 2>{{y0, y1}};
  }

 private:
  double a_, b_;
};

// Bad system of equations:
// f_1 (x0, x1) = a*x0**2 + b*x1**2 + c**2
// f_2 (x0, x1) = a*x0 + b*x1 + c
class BadFunction {
 public:
  BadFunction(const double& a, const double& b, const double& c)
      : a_(a), b_(b), c_(c) {}
  std::array<double, 2> operator()(const std::array<double, 2>& x) const
      noexcept {
    const double y0 = a_ * x[0] * x[0] + b_ * x[1] * x[1] + c_ * c_;
    const double y1 = a_ * x[0] + b_ * x[1] + c_;
    return std::array<double, 2>{{y0, y1}};
  }

 private:
  double a_, b_, c_;
};

template <typename Function>
void test_gsl_multiroot(RootFinder::StoppingCondition condition,
                        const Function& func,
                        const std::array<double, 2> initial_guess) {
  const double absolute_tolerance = 1.0e-14;
  const double max_absolute_tolerance = 0.0;
  const int maximum_iterations = 20;
  const RootFinder::Verbosity verbosity = RootFinder::Verbosity::Silent;

  std::vector<RootFinder::Method> methods_list{RootFinder::Method::Newton,
                                               RootFinder::Method::Hybrid,
                                               RootFinder::Method::Hybrids};
  for (const auto& method : methods_list) {
    double relative_tolerance = 1.0e-13;
    if (method == RootFinder::Method::Hybrids) {
      condition = RootFinder::StoppingCondition::Absolute;
      relative_tolerance = 0.0;
    }
    std::array<double, 2> roots_array = RootFinder::gsl_multiroot(
        func, initial_guess, absolute_tolerance, maximum_iterations,
        relative_tolerance, verbosity, max_absolute_tolerance, method,
        condition);
    CHECK(roots_array[0] == approx(1.0));
    CHECK(roots_array[1] == approx(1.0));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.GslMultiRoot",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  const std::array<double, 2> good_initial_guess{{-10.0, -5.0}};
  const Rosenbrock function_and_jac{1.0, 10.0};
  const RosenbrockNoJac function{1.0, 10.0};

  test_gsl_multiroot(RootFinder::StoppingCondition::AbsoluteAndRelative,
                     function_and_jac, good_initial_guess);
  test_gsl_multiroot(RootFinder::StoppingCondition::AbsoluteAndRelative,
                     function, good_initial_guess);
  test_gsl_multiroot(RootFinder::StoppingCondition::Absolute, function_and_jac,
                     good_initial_guess);
  test_gsl_multiroot(RootFinder::StoppingCondition::Absolute, function,
                     good_initial_guess);
  CHECK(get_output(RootFinder::Verbosity::Silent) == "Silent");
  CHECK(get_output(RootFinder::Verbosity::Quiet) == "Quiet");
  CHECK(get_output(RootFinder::Verbosity::Verbose) == "Verbose");
  CHECK(get_output(RootFinder::Verbosity::Debug) == "Debug");
  CHECK(get_output(RootFinder::Method::Hybrids) == "Hybrids");
  CHECK(get_output(RootFinder::Method::Hybrid) == "Hybrid");
  CHECK(get_output(RootFinder::Method::Newton) == "Newton");
  CHECK(get_output(RootFinder::StoppingCondition::AbsoluteAndRelative) ==
        "AbsoluteAndRelative");
  CHECK(get_output(RootFinder::StoppingCondition::Absolute) == "Absolute");

  test_throw_exception(
      []() {
        const std::array<double, 2> bad_initial_guess{{9.0e3, 2.0e5}};
        test_gsl_multiroot(RootFinder::StoppingCondition::AbsoluteAndRelative,
                           BadFunction{1.0, 1.0, 1.0}, bad_initial_guess);
      },
      convergence_error(
          "The root find failed and was not forgiven. An exception has been "
          "thrown.\n"
          "The gsl error returned is: the iteration has not converged yet\n"
          "Verbosity: Silent\n"
          "Method: Newton\n"
          "StoppingCondition: AbsoluteAndRelative\n"
          "Maximum absolute tolerance: 0\n"
          "Absolute tolerance: 1e-14\n"
          "Relative tolerance: 1e-13\n"
          "Maximum number of iterations: 20\n"
          "Number of iterations reached: 20\n"
          "The last value of f in the root solver is:\n"
          "1.74"));
}
