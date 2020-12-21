// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <pup.h>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/DampingFunction.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Overloader.hpp"

namespace TestHelpers::GeneralizedHarmonic::ConstraintDamping {
namespace detail {
template <size_t VolumeDim, typename Fr, class... MemberArgs, class T>
void check_impl(
    const std::unique_ptr<
        ::GeneralizedHarmonic::ConstraintDamping::DampingFunction<
            VolumeDim, Fr>>& in_gh_damping_function,
    const std::string& python_function_prefix, const T& used_for_size,
    const std::array<std::pair<double, double>, 1> random_value_bounds,
    const std::vector<std::string>& function_of_time_names,
    const MemberArgs&... member_args) noexcept {
  using GhDampingFunc =
      ::GeneralizedHarmonic::ConstraintDamping::DampingFunction<VolumeDim, Fr>;

  const auto member_args_tuple = std::make_tuple(member_args...);
  const auto helper =
      [&python_function_prefix, &random_value_bounds, &member_args_tuple,
       &function_of_time_names, &used_for_size](
          const std::unique_ptr<GhDampingFunc>& gh_damping_function) noexcept {
        INFO("Testing call operator...")
        // Make a lambda that calls the damping function's call operator
        // with a hard-coded FunctionsOfTime, since check_with_random_values
        // cannot convert a FunctionsOfTime into a python type.
        // The FunctionsOfTime contains a single FunctionOfTime
        // \f$f(t) = a_0 + a_1 (t-t_0) + a_2 (t-t_0)^2 + a_3 (t-t_0)^3\f$, where
        // \f$a_0 = 1.0\f$, \f$a_1 = 0.2\f$, \f$a_2 = 0.03,\f$,
        // \f$a_3 = 0.004\f$, and \f$t_0\f$ is the smallest possible value
        // of the randomly selected time.
        //
        // The corresponding python function should use
        // the same hard-coded coefficients to evaluate \f$f(t)\f$ as well
        // as the same value of \f$t_0\f$.
        // However, here the PiecewisePolynomial must be initialized not
        // with the polynomial coefficients but with the values of \f$f(t)\f$
        // and its derivatives evaluated at \f$t=t_0\f$: these are,
        // respectively, \f$a_0,a_1,2 a_2,6 a_3\f$.
        //
        // Finally, note that the FunctionOfTime never expires.
        const auto damping_function_call_operator_helper =
            [&gh_damping_function, &random_value_bounds,
             &function_of_time_names](
                const tnsr::I<T, VolumeDim, Fr>& coordinates,
                const double time) {
              std::unordered_map<
                  std::string,
                  std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>
                  functions_of_time{};
              for (auto function_of_time_name : function_of_time_names) {
                // The randomly selected time will be between the
                // random_value_bounds, so set the earliest time of the
                // function_of_times to the lower bound in random_value_bounds.
                functions_of_time[function_of_time_name] = std::make_unique<
                    ::domain::FunctionsOfTime::PiecewisePolynomial<3>>(
                    std::min(gsl::at(random_value_bounds, 0).first,
                             gsl::at(random_value_bounds, 0).second),
                    std::array<DataVector, 4>{{{1.0}, {0.2}, {0.06}, {0.024}}},
                    std::numeric_limits<double>::max());
              }
              // Default-construct the scalar, to test that the damping
              // function's call operator correctly resizes it
              // (in the case T is a DataVector) with
              // destructive_resize_components()
              Scalar<T> value_at_coordinates{};
              gh_damping_function->operator()(
                  make_not_null(&value_at_coordinates), coordinates, time,
                  functions_of_time);
              return value_at_coordinates;
            };

        pypp::check_with_random_values<1>(
            &decltype(damping_function_call_operator_helper)::operator(),
            damping_function_call_operator_helper, "TestFunctions",
            python_function_prefix + "_call_operator", random_value_bounds,
            member_args_tuple, used_for_size);
        INFO("Done testing call operator...")
        INFO("Done\n\n")
      };

  helper(in_gh_damping_function);
  helper(serialize_and_deserialize(in_gh_damping_function));
}
}  // namespace detail
// @{
/*!
 * \ingroup TestingFrameworkGroup
 * \brief Test a DampingFunction by comparing to python functions
 *
 * The python functions must be added to TestFunctions.py in
 * tests/Unit/Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Python.
 * Each python function for a corresponding DampingFunction should begin
 * with a prefix `python_function_prefix`. The prefix for each class of
 * DampingFunction is arbitrary, but should generally be descriptive (e.g.
 * 'gaussian_plus_constant') of the DampingFunction.
 *
 * The input parameter `function_of_time_name` is the name of the FunctionOfTime
 * that will be included in the FunctionsOfTime passed to the DampingFunction's
 * call operator. For time-dependent DampingFunctions, this parameter must be
 * consistent with the FunctionOfTime name that the call operator of
 * `in_gh_damping_function` expects. For time-independent DampingFunctions,
 * `function_of_time_name` will be ignored.
 *
 * If a DampingFunction class has member variables set by its constructor, then
 * these member variables must be passed in as the last arguments to the `check`
 * function`. Each python function must take these same arguments as the
 * trailing arguments.
 */
template <class DampingFunctionType, class T, class... MemberArgs>
void check(std::unique_ptr<DampingFunctionType> in_gh_damping_function,
           const std::string& python_function_prefix, const T& used_for_size,
           const std::array<std::pair<double, double>, 1>& random_value_bounds,
           const std::vector<std::string>& function_of_time_names,
           const MemberArgs&... member_args) noexcept {
  detail::check_impl(
      std::unique_ptr<::GeneralizedHarmonic::ConstraintDamping::DampingFunction<
          DampingFunctionType::volume_dim,
          typename DampingFunctionType::frame>>(
          std::move(in_gh_damping_function)),
      python_function_prefix, used_for_size, random_value_bounds,
      function_of_time_names, member_args...);
}

template <class DampingFunctionType, class T, class... MemberArgs>
void check(DampingFunctionType in_gh_damping_function,
           const std::string& python_function_prefix, const T& used_for_size,
           const std::array<std::pair<double, double>, 1>& random_value_bounds,
           const std::vector<std::string>& function_of_time_names,
           const MemberArgs&... member_args) noexcept {
  detail::check_impl(
      std::unique_ptr<::GeneralizedHarmonic::ConstraintDamping::DampingFunction<
          DampingFunctionType::volume_dim,
          typename DampingFunctionType::frame>>(
          std::make_unique<DampingFunctionType>(
              std::move(in_gh_damping_function))),
      python_function_prefix, used_for_size, random_value_bounds,
      function_of_time_names, member_args...);
}
// @}
}  // namespace TestHelpers::GeneralizedHarmonic::ConstraintDamping
