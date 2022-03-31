// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <type_traits>
#include <utility>

#include "DataStructures/MathWrapper.hpp"
#include "Helpers/DataStructures/MathWrapperDetail.hpp"

namespace TestHelpers::MathWrapper {

template <typename T, typename Scalar>
void test_type(const T& first_value, const T& second_value,
               const Scalar& scalar) {
  T destination = first_value;
  const auto destination_wrapper =
      make_math_wrapper(make_not_null(&destination));

  T source = second_value;
  const auto mutable_source_wrapper = make_math_wrapper(make_not_null(&source));
  const auto const_source_wrapper = make_math_wrapper(std::as_const(source));
  using MutableWrapper = std::decay_t<decltype(mutable_source_wrapper)>;
  using ConstWrapper = std::decay_t<decltype(const_source_wrapper)>;

  // Test value_type
  static_assert(not std::is_const_v<typename MutableWrapper::value_type>);
  static_assert(std::is_const_v<typename ConstWrapper::value_type>);
  static_assert(
      std::is_same_v<::MathWrapper<typename MutableWrapper::value_type>,
                     MutableWrapper>);
  static_assert(std::is_same_v<::MathWrapper<typename ConstWrapper::value_type>,
                               ConstWrapper>);
  static_assert(std::is_same_v<math_wrapper_type<T>,
                               typename MutableWrapper::value_type>);
  static_assert(std::is_same_v<math_wrapper_type<const T>,
                               typename ConstWrapper::value_type>);

  detail::do_assignment(destination_wrapper, mutable_source_wrapper.to_const());
  CHECK(destination == second_value);
  CHECK(source == second_value);
  source = first_value;
  detail::do_assignment(destination_wrapper, mutable_source_wrapper.to_const());
  CHECK(destination == first_value);
  CHECK(source == first_value);
  source = second_value;
  detail::do_assignment(destination_wrapper, const_source_wrapper);
  CHECK(destination == second_value);
  CHECK(source == second_value);
  source = first_value;
  detail::do_assignment(destination_wrapper, const_source_wrapper);
  CHECK(destination == first_value);
  CHECK(source == first_value);

  static_assert(std::is_same_v<typename MutableWrapper::scalar_type, Scalar>);
  static_assert(std::is_same_v<typename ConstWrapper::scalar_type, Scalar>);
  static_assert(std::is_same_v<Scalar, double> or
                std::is_same_v<Scalar, std::complex<double>>);
  source = first_value;
  detail::do_multiply(destination_wrapper, scalar,
                      mutable_source_wrapper.to_const());
  CHECK(destination == T(scalar * first_value));
  source = second_value;
  detail::do_multiply(destination_wrapper, scalar, const_source_wrapper);
  CHECK(destination == T(scalar * second_value));
}
}  // namespace TestHelpers::MathWrapper
