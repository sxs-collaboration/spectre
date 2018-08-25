// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_vector_double.h>
#include <ostream>
#include <string>

#include "ErrorHandling/Error.hpp"
#include "ErrorHandling/Exceptions.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"

namespace RootFinder {
namespace gsl_multiroot_detail {
template <size_t Dim, typename Solver>
void print_state(const size_t iteration_number, const Solver* const solver,
                 const bool print_header = false) noexcept {
  if (print_header) {
    Parallel::printf("Iter\t");
    for (size_t i = 0; i < Dim; ++i) {
      Parallel::printf(" x[%u]\t", i);
    }
    for (size_t i = 0; i < Dim; ++i) {
      Parallel::printf(" f[%u]\t", i);
    }
    Parallel::printf("\n");
  }

  Parallel::printf("%u\t", iteration_number);
  for (size_t i = 0; i < Dim; ++i) {
    Parallel::printf("%3.4f  ", gsl_vector_get(solver->x, i));
  }
  for (size_t i = 0; i < Dim; ++i) {
    Parallel::printf("%1.3e  ", gsl_vector_get(solver->f, i));
  }
  Parallel::printf("\n");
}

template <size_t Dim>
std::array<double, Dim> gsl_to_std_array(const gsl_vector* const x) noexcept {
  std::array<double, Dim> input_as_std_array{};
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(input_as_std_array, i) = gsl_vector_get(x, i);
  }
  return input_as_std_array;
}

template <size_t Dim>
void gsl_vector_set_with_std_array(
    gsl_vector* const func,
    const std::array<double, Dim>& result_as_std_array) noexcept {
  for (size_t i = 0; i < Dim; i++) {
    gsl_vector_set(func, i, gsl::at(result_as_std_array, i));
  }
}

template <size_t Dim>
void gsl_matrix_set_with_std_array(
    gsl_matrix* const matrix,
    const std::array<std::array<double, Dim>, Dim>& matrix_array) noexcept {
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      gsl_matrix_set(matrix, i, j, gsl::at(gsl::at(matrix_array, i), j));
    }
  }
}

// The gsl_multiroot_function_fdf expects its functions to be of the form
// int (* f) (const gsl_vector * x, void * params, gsl_vector * f).
// However, we would like to be able to perform rootfinding on functions
// of the form std::array<double, Dim> f(const std::array<double, Dim>& x).
// So, we pass the function wrapper below to gsl_multiroot_function_fdf.
// In the gsl documentation the third parameter is refered to as "void* params",
// referring to the parameters that select out a particular function out of a
// family of possible functions described in a class, but we instead pass the
// pointer to the entire function object itself here. The type of the function
// object is passed through with the Function template parameter.
template <size_t Dim, typename Function>
int gsl_multirootfunctionfdf_wrapper_f(const gsl_vector* const x,
                                       void* const untyped_function_object,
                                       gsl_vector* const f_of_x) noexcept {
  const auto function_object =
      static_cast<const Function*>(untyped_function_object);
  gsl_vector_set_with_std_array(
      f_of_x, function_object->operator()(gsl_to_std_array<Dim>(x)));
  return GSL_SUCCESS;
}

template <size_t Dim, typename Function>
int gsl_multirootfunctionfdf_wrapper_df(const gsl_vector* const x,
                                        void* const untyped_function_object,
                                        gsl_matrix* const jacobian) noexcept {
  const auto function_object =
      static_cast<const Function*>(untyped_function_object);
  gsl_matrix_set_with_std_array(
      jacobian, function_object->jacobian(gsl_to_std_array<Dim>(x)));

  return GSL_SUCCESS;
}

template <size_t Dim, typename Function>
int gsl_multirootfunctionfdf_wrapper_fdf(const gsl_vector* const x,
                                         void* const untyped_function_object,
                                         gsl_vector* const f_of_x,
                                         gsl_matrix* const jacobian) noexcept {
  const auto function_object =
      static_cast<const Function*>(untyped_function_object);
  const std::array<double, Dim> x_as_std_array = gsl_to_std_array<Dim>(x);
  gsl_vector_set_with_std_array(f_of_x,
                                function_object->operator()(x_as_std_array));
  gsl_matrix_set_with_std_array(jacobian,
                                function_object->jacobian(x_as_std_array));
  return GSL_SUCCESS;
}

CREATE_IS_CALLABLE(jacobian)
}  // namespace gsl_multiroot_detail

/*!
 *  \ingroup NumericalAlgorithmsGroup
 *  \brief The different options for the verbosity of gsl_multiroot.
 */
enum class Verbosity {
  /// Do not print anything
  Silent,
  /// Print only "success" or "failed" on termination
  Quiet,
  /// Print final functions values on termination
  Verbose,
  /// Print function values on every iteration
  Debug
};

std::ostream& operator<<(std::ostream& /*os*/,
                         const Verbosity& /*verbosity*/) noexcept;

/*!
 *  \ingroup NumericalAlgorithmsGroup
 *  \brief The different options for the rootfinding method of gsl_multiroot.
 *
 *  This enum is for setting the method used the rootfinder.
 *  The precise method used by the gsl rootfinder depends on whether or not the
 *  function passed to it has a callable `jacobian` member function. In the
 *  case where it doesn't, the jacobian is approximated with a finite
 *  difference. For example, if the Method specified is Hybrid, gsl will use
 *  the gsl_multiroot_fdfsolver_hybridj method in the case where a `jacobian`
 *  is provided, and gsl_multiroot_fsolver_hybrid in the case where one isn't.
 *  See
 *  [GSL's documentation for multidimensional
 *  rootfinding](https://www.gnu.org/software/gsl/manual/html_node/Multidimensional-Root_002dFinding.html)
 *  for information on the different methods.
 *  \note gsl does not provide a finite difference version for the modified
 *  Newton method (gsl_multiroot_fdfsolver_gnewton). In the case where a
 *  jacobian is not provided the method used will be a non-modified Newton
 *  method.
 *
 */
enum class Method {
  /// Hybrid of Newton's method along with following the gradient direction.
  /// \note Sometimes Hybrids works only with the Absolute stopping condition.
  Hybrids,
  /// "Unscaled version of Hybrids that uses a spherical trust region," see
  /// GSL documentation for more details.
  Hybrid,
  /// If an analytic jacobian is provided, gsl uses a modification of Newton's
  /// method to improve global convergence. Uses vanilla Newton's method if no
  /// jacobian is provided.
  Newton
};

std::ostream& operator<<(std::ostream& /*os*/,
                         const Method& /*method*/) noexcept;

/*!
 *  \ingroup NumericalAlgorithmsGroup
 *  \brief The different options for the convergence criterion of gsl_multiroot.
 *
 *  See
 *  [GSL's documentation for multidimensional
 *  rootfinding](https://www.gnu.org/software/gsl/manual/html_node/Multidimensional-Root_002dFinding.html)
 *  for information on the different stopping conditions.
 */
enum class StoppingCondition {
  /// See GSL documentation for gsl_multiroot_test_delta.
  AbsoluteAndRelative,
  /// See GSL documentation for gsl_multiroot_test_residual.
  Absolute
};

std::ostream& operator<<(std::ostream& /*os*/,
                         const StoppingCondition& /*condition*/) noexcept;

namespace gsl_multiroot_detail {
template <typename SolverType, typename SolverAlloc, typename SolverSet,
          typename SolverIterate, typename SolverFree, size_t Dim,
          typename Function>
std::array<double, Dim> gsl_multiroot_impl(
    Function& f, const std::array<double, Dim>& initial_guess,
    const double absolute_tolerance, const size_t maximum_iterations,
    const double relative_tolerance, const Verbosity verbosity,
    const double maximum_absolute_tolerance, const Method method,
    const SolverType solver_type, const StoppingCondition condition,
    const SolverAlloc solver_alloc, const SolverSet solver_set,
    const SolverIterate solver_iterate, const SolverFree solver_free) {
  // Check for valid stopping condition:
  if (UNLIKELY(condition != StoppingCondition::AbsoluteAndRelative and
               condition != StoppingCondition::Absolute)) {
    ERROR(
        "Invalid stopping condition. Has to be either AbsoluteAndRelative"
        "or Absolute.");
  }

  // Supply gsl_root with the initial guess:
  gsl_vector* const gsl_root = gsl_vector_alloc(Dim);
  gsl_vector_set_with_std_array(gsl_root, initial_guess);
  auto* const solver = solver_alloc(solver_type, Dim);
  solver_set(solver, &f, gsl_root);

  // Take iterations:
  int status;
  size_t iteration_number = 0;
  do {
    if (UNLIKELY(verbosity == Verbosity::Debug)) {
      print_state<Dim>(iteration_number, solver, iteration_number == 0);
    }
    iteration_number++;
    status = solver_iterate(solver);
    // Check if solver is stuck
    if (UNLIKELY(status == GSL_ENOPROG)) {
      if (UNLIKELY(verbosity == Verbosity::Debug)) {
        Parallel::printf(
            "The iteration is not making any progress, preventing the "
            "algorithm from continuing.");
      }
      break;
    }
    if (condition == StoppingCondition::AbsoluteAndRelative) {
      status = gsl_multiroot_test_delta(solver->dx, solver->x,
                                        absolute_tolerance, relative_tolerance);
    } else {  // condition is StoppingCondition::Absolute
      // NOTE: Sometimes hybridsj works only with the test_residual condition
      status = gsl_multiroot_test_residual(solver->f, absolute_tolerance);
    }
  } while (status == GSL_CONTINUE and iteration_number < maximum_iterations);
  if (UNLIKELY(verbosity == Verbosity::Verbose or
               verbosity == Verbosity::Debug)) {
    Parallel::printf("Finished iterating:\n");
    print_state<Dim>(iteration_number, solver, verbosity == Verbosity::Verbose);
  }
  bool success = (status == GSL_SUCCESS);
  if (UNLIKELY(verbosity != Verbosity::Silent)) {
    Parallel::printf("\n");
    if (not success) {
      const std::string ascii_divider = std::string(70, '#');
      const std::string failure_message =
          ascii_divider + "\n\t\tWARNING: Root Finding FAILED\n" +
          ascii_divider;
      Parallel::printf("%s\n", failure_message);
    } else {
      Parallel::printf("Root finder converged.\n");
    }
  }
  // If maximum_absolute_tolerance is given, return success = true
  // as long as maximum_absolute_tolerance is achieved even if the
  // root finder doesn't converge.
  bool success_with_tolerance = true;
  bool failed_root_is_forgiven = false;
  if (not success and maximum_absolute_tolerance > 0.0) {
    for (size_t i = 0; i < Dim; ++i) {
      if (fabs(gsl_vector_get(solver->f, i)) > maximum_absolute_tolerance) {
        success_with_tolerance = false;
      }
    }
    failed_root_is_forgiven = success_with_tolerance;
  }
  if (success_with_tolerance and maximum_absolute_tolerance > 0.0) {
    success = true;
  }
  if (UNLIKELY(failed_root_is_forgiven and (verbosity == Verbosity::Verbose or
                                            verbosity == Verbosity::Debug))) {
    Parallel::printf(
        "The failed root was forgiven as each component was found to be under "
        "maximum_absolute_tolerance %f",
        maximum_absolute_tolerance);
  }

  if (UNLIKELY(not success)) {
    std::stringstream error_message;
    error_message << "The root find failed and was not forgiven. An exception "
                     "has been thrown.\n"
                  << "The gsl error returned is: " << gsl_strerror(status)
                  << "\n"
                  << "Verbosity: " << verbosity << "\n"
                  << "Method: " << method << "\n"
                  << "StoppingCondition: " << condition << "\n"
                  << "Maximum absolute tolerance: "
                  << maximum_absolute_tolerance << "\n"
                  << "Absolute tolerance: " << absolute_tolerance << "\n"
                  << "Relative tolerance: " << relative_tolerance << "\n"
                  << "Maximum number of iterations: " << maximum_iterations
                  << "\n"
                  << "Number of iterations reached: " << iteration_number
                  << "\n"
                  << "The last value of f in the root solver is:\n";
    for (size_t i = 0; i < Dim; i++) {
      error_message << gsl_vector_get(solver->f, i) << "\n";
    }
    error_message << "The last value of x in the root solver is:\n";
    for (size_t i = 0; i < Dim; i++) {
      error_message << gsl_vector_get(solver->x, i) << "\n";
    }
    error_message << "The last value of dx in the root solver is:\n";
    for (size_t i = 0; i < Dim; i++) {
      error_message << gsl_vector_get(solver->dx, i) << "\n";
    }

    if (UNLIKELY(verbosity == Verbosity::Debug)) {
      Parallel::printf("Error: %s\n", gsl_strerror(status));
    if (iteration_number >= maximum_iterations) {
      Parallel::printf(
          "The number of iterations (%d) has reached the maximum number of "
          "iterations (%d)\n",
          iteration_number, maximum_iterations);
    } else {
      Parallel::printf(
          "The number of iterations (%d) failed to reach the maximum number of "
          "iterations (%d)\n",
          iteration_number, maximum_iterations);
    }
    }
    throw convergence_error(error_message.str());
  }

  // Store the converged root in result
  std::array<double, Dim> result = gsl_to_std_array<Dim>(solver->x);
  solver_free(solver);
  gsl_vector_free(gsl_root);

  return result;
}

void print_rootfinding_parameters(Method method, double absolute_tolerance,
                                  double relative_tolerance,
                                  double maximum_absolute_tolerance,
                                  StoppingCondition condition) noexcept;
}  // namespace gsl_multiroot_detail

// @{
/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief A multidimensional root finder supporting Newton and Hybrid
 * methods, as well as modified methods based on these.
 *
 * This root finder accepts function objects with and without a callable
 * `jacobian` member function. This member function both accepts and
 * returns a `std::array<double, Dim>`, the dimension of the domain and range
 * of the function the root find is being performed on. Whether the jacobian
 * is provided determines the details of the implementation of the
 * root-finding method that is selected by the user using the Method enum.
 * That is, whether the jacobian is computed analytically via the `jacobian`
 * member function, or whether the jacobian is computed numerically via a
 * finite difference approximation.
 * \note GSL does not provide a finite difference version of its modified
 * Newton method, so the unmodified one is used instead when the user
 * uses the Method::Newton method.
 *
 * The user can select one of two possible criteria for convergence,
 * StoppingCondition::Absolute, where the sum of the absolute values of the
 * components of the residual vector f are compared against the value
 * provided to `absolute_tolerance`, and
 * StoppingCondition::AbsoluteAndRelative, where the size of the most recent
 * step taken in the root-finding iteration is compared against
 * `absolute_tolerance` + `relative_tolerance` * |x_i|, for each component.
 * In either case, a `maximum_absolute_tolerance` may be specified if the user
 * anticipates that the convergence criterion specified with StoppingCondition
 * will be too strict for a few points out of a population of points found with
 * a sequence of root finds.
 *
 * See
 * [GSL's documentation for multidimensional
 * rootfinding](https://www.gnu.org/software/gsl/manual/html_node/Multidimensional-Root_002dFinding.html)
 * for reference.
 *
 * \param func Function whose root is to be found.
 * \param initial_guess Contains initial guess.
 * \param absolute_tolerance The absolute tolerance.
 * \param maximum_iterations The maximum number of iterations.
 * \param relative_tolerance The relative tolerance.
 * \param verbosity Whether to print diagnostic messages.
 * \param maximum_absolute_tolerance Acceptable absolute tolerance when
 *                                   root finder doesn't converge.
 *                                   You may wish to use this if there
 *                                   are only a few "problematic" points where
 *                                   it is difficult to do a precise root find.
 * \param method The method to use. See the documentation for the Method enum.
 * \param condition The convergence condition to use. See the documentation
 *                                   for the StoppingCondition enum.
 */
template <size_t Dim, typename Function,
          Requires<gsl_multiroot_detail::is_jacobian_callable_v<
              Function, std::array<double, Dim>>> = nullptr>
std::array<double, Dim> gsl_multiroot(
    const Function& func, const std::array<double, Dim>& initial_guess,
    const double absolute_tolerance, const size_t maximum_iterations,
    const double relative_tolerance = 0.0,
    const Verbosity verbosity = Verbosity::Silent,
    const double maximum_absolute_tolerance = 0.0,
    const Method method = Method::Newton,
    const StoppingCondition condition = StoppingCondition::Absolute) {
  gsl_multiroot_function_fdf gsl_func = {
      &gsl_multiroot_detail::gsl_multirootfunctionfdf_wrapper_f<Dim, Function>,
      &gsl_multiroot_detail::gsl_multirootfunctionfdf_wrapper_df<Dim, Function>,
      &gsl_multiroot_detail::gsl_multirootfunctionfdf_wrapper_fdf<Dim,
                                                                  Function>,
      Dim, const_cast<Function*>(&func)};  //NOLINT

  // Set up method for solver:
  const gsl_multiroot_fdfsolver_type* solver_type;
  if (method == Method::Newton) {
    solver_type = gsl_multiroot_fdfsolver_gnewton;
  } else if (method == Method::Hybrids) {
    solver_type = gsl_multiroot_fdfsolver_hybridsj;
  } else if (method == Method::Hybrid) {
    solver_type = gsl_multiroot_fdfsolver_hybridj;
  } else {
    ERROR(
        "Invalid method. Has to be one of Newton, Hybrids or "
        "Hybrid.");
  }
  // Print initial parameters
  if (UNLIKELY(verbosity == Verbosity::Verbose or
               verbosity == Verbosity::Debug)) {
    gsl_multiroot_detail::print_rootfinding_parameters(
        method, absolute_tolerance, relative_tolerance,
        maximum_absolute_tolerance, condition);
  }
  return gsl_multiroot_detail::gsl_multiroot_impl(
      gsl_func, initial_guess, absolute_tolerance, maximum_iterations,
      relative_tolerance, verbosity, maximum_absolute_tolerance, method,
      solver_type, condition, &gsl_multiroot_fdfsolver_alloc,
      &gsl_multiroot_fdfsolver_set, &gsl_multiroot_fdfsolver_iterate,
      &gsl_multiroot_fdfsolver_free);
}

template <size_t Dim, typename Function,
          Requires<not gsl_multiroot_detail::is_jacobian_callable_v<
              Function, std::array<double, Dim>>> = nullptr>
std::array<double, Dim> gsl_multiroot(
    const Function& func, const std::array<double, Dim>& initial_guess,
    const double absolute_tolerance, const size_t maximum_iterations,
    const double relative_tolerance = 0.0,
    const Verbosity verbosity = Verbosity::Silent,
    const double maximum_absolute_tolerance = 0.0,
    const Method method = Method::Newton,
    const StoppingCondition condition = StoppingCondition::Absolute) {
  gsl_multiroot_function gsl_func = {
      &gsl_multiroot_detail::gsl_multirootfunctionfdf_wrapper_f<Dim, Function>,
      Dim, const_cast<Function*>(&func)};  // NOLINT

  // Set up method for solver:
  const gsl_multiroot_fsolver_type* solver_type;
  if (method == Method::Newton) {
    solver_type = gsl_multiroot_fsolver_dnewton;
  } else if (method == Method::Hybrids) {
    solver_type = gsl_multiroot_fsolver_hybrids;
  } else if (method == Method::Hybrid) {
    solver_type = gsl_multiroot_fsolver_hybrid;
  } else {
    ERROR(
        "Invalid method. Has to be one of Newton, Hybrids or "
        "Hybrid.");
  }
  // Print initial parameters
  if (UNLIKELY(verbosity == Verbosity::Verbose or
               verbosity == Verbosity::Debug)) {
    gsl_multiroot_detail::print_rootfinding_parameters(
        method, absolute_tolerance, relative_tolerance,
        maximum_absolute_tolerance, condition);
  }
  return gsl_multiroot_detail::gsl_multiroot_impl(
      gsl_func, initial_guess, absolute_tolerance, maximum_iterations,
      relative_tolerance, verbosity, maximum_absolute_tolerance, method,
      solver_type, condition, &gsl_multiroot_fsolver_alloc,
      &gsl_multiroot_fsolver_set, &gsl_multiroot_fsolver_iterate,
      &gsl_multiroot_fsolver_free);
}
// @}
}  // namespace RootFinder
