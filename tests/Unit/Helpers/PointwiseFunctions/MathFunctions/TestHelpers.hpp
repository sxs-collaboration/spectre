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
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Overloader.hpp"

namespace TestHelpers {
namespace MathFunctions {
namespace detail {
template <size_t VolumeDim, typename Fr, class... MemberArgs, class T>
void check_impl(
    const std::unique_ptr<MathFunction<VolumeDim, Fr>>& in_math_function,
    const std::string& python_function_prefix, const T& used_for_size,
    const std::array<std::pair<double, double>, 1> random_value_bounds,
    const MemberArgs&... member_args) noexcept {
  using MathFunc = MathFunction<VolumeDim, Fr>;
  using CallOperatorFunction =
      Scalar<T> (MathFunc::*)(const tnsr::I<T, VolumeDim, Fr>&) const noexcept;
  using FirstDerivFunction =
      tnsr::i<T, VolumeDim, Fr> (MathFunc::*)(const tnsr::I<T, VolumeDim, Fr>&)
          const noexcept;
  using SecondDerivFunction =
      tnsr::ii<T, VolumeDim, Fr> (MathFunc::*)(const tnsr::I<T, VolumeDim, Fr>&)
          const noexcept;
  using ThirdDerivFunction = tnsr::iii<T, VolumeDim, Fr> (MathFunc::*)(
      const tnsr::I<T, VolumeDim, Fr>&) const noexcept;

  const auto member_args_tuple = std::make_tuple(member_args...);
  const auto helper =
      [&](const std::unique_ptr<MathFunc>& math_function) noexcept {
        // need func variable to work around GCC bug
        CallOperatorFunction func{&MathFunc::operator()};

        INFO("Testing call operator...")
        pypp::check_with_random_values<1>(
            func, *math_function, "TestFunctions",
            python_function_prefix + "_call_operator", random_value_bounds,
            member_args_tuple, used_for_size);
        INFO("Done testing call operator...")

        FirstDerivFunction d_func{&MathFunc::first_deriv};
        INFO("Testing first derivative...")
        pypp::check_with_random_values<1>(
            d_func, *math_function, "TestFunctions",
            python_function_prefix + "_first_deriv", random_value_bounds,
            member_args_tuple, used_for_size);
        INFO("Done testing first derivative...")

        SecondDerivFunction d2_func{&MathFunc::second_deriv};
        INFO("Testing second derivative...")
        pypp::check_with_random_values<1>(
            d2_func, *math_function, "TestFunctions",
            python_function_prefix + "_second_deriv", random_value_bounds,
            member_args_tuple, used_for_size);
        INFO("Done testing second derivative...")

        ThirdDerivFunction d3_func{&MathFunc::third_deriv};
        INFO("Testing third derivative...")
        pypp::check_with_random_values<1>(
            d3_func, *math_function, "TestFunctions",
            python_function_prefix + "_third_deriv", random_value_bounds,
            member_args_tuple, used_for_size);
        INFO("Done testing third derivative...")

        INFO("Done\n\n")
      };
  helper(in_math_function);
  helper(serialize_and_deserialize(in_math_function));

  if constexpr (VolumeDim == 1) {
    // Check that the tensor interface agrees with the double/DataVector
    // interface

    MAKE_GENERATOR(gen);
    std::uniform_real_distribution<> real_dis(-1, 1);
    const auto nn_generator = make_not_null(&gen);
    const auto nn_distribution = make_not_null(&real_dis);

    auto coords_tensor = make_with_random_values<tnsr::I<T, VolumeDim, Fr>>(
        nn_generator, nn_distribution, used_for_size);
    T coords_T = get<0>(coords_tensor);

    CHECK_ITERABLE_APPROX(get((*in_math_function)(coords_tensor)),
                          (*in_math_function)(coords_T));

    const T deriv_from_tensor =
        std::move(get<0>(in_math_function->first_deriv(coords_tensor)));
    CHECK_ITERABLE_APPROX(deriv_from_tensor,
                          in_math_function->first_deriv(coords_T));

    const T second_deriv_from_tensor =
        std::move(get<0, 0>(in_math_function->second_deriv(coords_tensor)));
    CHECK_ITERABLE_APPROX(second_deriv_from_tensor,
                          in_math_function->second_deriv(coords_T));

    const T third_deriv_from_tensor =
        std::move(get<0, 0, 0>(in_math_function->third_deriv(coords_tensor)));
    CHECK_ITERABLE_APPROX(third_deriv_from_tensor,
                          in_math_function->third_deriv(coords_T));
  }
}
}  // namespace detail
// @{
/*!
 * \ingroup TestingFrameworkGroup
 * \brief Test a MathFunction by comparing to python functions
 *
 * The python functions must be added to
 * tests/Unit/PointwiseFunctions/MathFunctions/Python/TestFunctions.py. The
 * prefix for each class of MathFunction is arbitrary, but should generally
 * be descriptive (e.g. 'gaussian', 'sinusoid', 'pow_x') of the MathFunction.
 *
 * The `python_function_prefix` argument passed to `check` must be `PREFIX`. If
 * a MathFunction class has member variables set by its constructor, then these
 * member variables must be passed in as the last arguments to the `check`
 * function`. Each python function must take these same arguments as the
 * trailing arguments.
 */
template <class MathFunctionType, class T, class... MemberArgs>
void check(std::unique_ptr<MathFunctionType> in_math_function,
           const std::string& python_function_prefix, const T& used_for_size,
           const std::array<std::pair<double, double>, 1> random_value_bounds,
           const MemberArgs&... member_args) noexcept {
  detail::check_impl(
      std::unique_ptr<MathFunction<MathFunctionType::volume_dim,
                                   typename MathFunctionType::frame>>(
          std::move(in_math_function)),
      python_function_prefix, used_for_size, random_value_bounds,
      member_args...);
}

template <class MathFunctionType, class T, class... MemberArgs>
void check(MathFunctionType in_math_function,
           const std::string& python_function_prefix, const T& used_for_size,
           const std::array<std::pair<double, double>, 1> random_value_bounds,
           const MemberArgs&... member_args) noexcept {
  detail::check_impl(
      std::unique_ptr<MathFunction<MathFunctionType::volume_dim,
                                   typename MathFunctionType::frame>>(
          std::make_unique<MathFunctionType>(std::move(in_math_function))),
      python_function_prefix, used_for_size, random_value_bounds,
      member_args...);
}
// @}
}  // namespace MathFunctions
}  // namespace TestHelpers
