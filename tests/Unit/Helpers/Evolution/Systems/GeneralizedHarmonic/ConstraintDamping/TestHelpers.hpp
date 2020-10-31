// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <pup.h>
#include <string>
#include <tuple>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
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
      ::GeneralizedHarmonic::ConstraintDamping::DampingFunction<VolumeDim,
                                                                  Fr>;
  using CallOperatorFunction =
      Scalar<T> (GhDampingFunc::*)(const tnsr::I<T, VolumeDim, Fr>&)
          const noexcept;

  const auto member_args_tuple = std::make_tuple(member_args...);
  const auto helper =
      [&python_function_prefix, &random_value_bounds, &member_args_tuple,
       &used_for_size](
          const std::unique_ptr<GhDampingFunc>& gh_damping_function) noexcept {
        INFO("Testing call operator...")
        pypp::check_with_random_values<1>(
            static_cast<CallOperatorFunction>(&GhDampingFunc::operator()),
            *gh_damping_function, "TestFunctions",
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
           const std::array<std::pair<double, double>, 1> random_value_bounds,
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
           const std::array<std::pair<double, double>, 1> random_value_bounds,
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
