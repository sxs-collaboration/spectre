// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <array>
#include <limits>
#include <memory>
#include <pup.h>
#include <string>
#include <tuple>
#include <utility>

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
    const MemberArgs&... member_args) noexcept {
  using GhDampingFunc =
      ::GeneralizedHarmonic::ConstraintDamping::DampingFunction<VolumeDim, Fr>;

  const auto member_args_tuple = std::make_tuple(member_args...);
  const auto helper =
      [&python_function_prefix, &random_value_bounds, &member_args_tuple,
       &used_for_size](
          const std::unique_ptr<GhDampingFunc>& gh_damping_function) noexcept {
        INFO("Testing call operator...")

        // Make a lambda that calls the damping function's call operator
        // with a hard-coded FunctionsOfTime, since check_with_random_values
        // cannot convert a FunctionsOfTime into a python type.
        // The FunctionsOfTime contains a single FunctionOfTime
        // \f$f(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3\f$, where
        // \f$a_0 = 1.0\f$, \f$a_1 = 0.2\f$, \f$a_3 = 0.03,\f$ and
        // \f$a_4 = 0.004\f$. The corresponding python function should use
        // the same hard-coded coefficients to evaluate \f$f(t)\f$.
        // Note that the FunctionOfTime never expires.
        const auto damping_function_call_operator_helper =
            [&gh_damping_function](const tnsr::I<T, VolumeDim, Fr>& coordinates,
                                   const double time) {
              std::unordered_map<
                  std::string,
                  std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>
                  functions_of_time{};
              functions_of_time["ExpansionFactor"] = std::make_unique<
                  ::domain::FunctionsOfTime::PiecewisePolynomial<3>>(
                  0.0,
                  std::array<DataVector, 4>{{{1.0}, {0.2}, {0.03}, {0.004}}},
                  std::numeric_limits<double>::max());
              return gh_damping_function->operator()(coordinates, time,
                                                     functions_of_time);
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
 * If a DampingFunction class has member variables set by its constructor, then
 * these member variables must be passed in as the last arguments to the `check`
 * function`. Each python function must take these same arguments as the
 * trailing arguments.
 */
template <class DampingFunctionType, class T, class... MemberArgs>
void check(std::unique_ptr<DampingFunctionType> in_gh_damping_function,
           const std::string& python_function_prefix, const T& used_for_size,
           const std::array<std::pair<double, double>, 1>& random_value_bounds,
           const MemberArgs&... member_args) noexcept {
  detail::check_impl(
      std::unique_ptr<
          ::GeneralizedHarmonic::ConstraintDamping::DampingFunction<
              DampingFunctionType::volume_dim,
              typename DampingFunctionType::frame>>(
          std::move(in_gh_damping_function)),
      python_function_prefix, used_for_size, random_value_bounds,
      member_args...);
}

template <class DampingFunctionType, class T, class... MemberArgs>
void check(DampingFunctionType in_gh_damping_function,
           const std::string& python_function_prefix, const T& used_for_size,
           const std::array<std::pair<double, double>, 1>& random_value_bounds,
           const MemberArgs&... member_args) noexcept {
  detail::check_impl(
      std::unique_ptr<
          ::GeneralizedHarmonic::ConstraintDamping::DampingFunction<
              DampingFunctionType::volume_dim,
              typename DampingFunctionType::frame>>(
          std::make_unique<DampingFunctionType>(
              std::move(in_gh_damping_function))),
      python_function_prefix, used_for_size, random_value_bounds,
      member_args...);
}
// @}
}  // namespace TestHelpers::GeneralizedHarmonic::ConstraintDamping
