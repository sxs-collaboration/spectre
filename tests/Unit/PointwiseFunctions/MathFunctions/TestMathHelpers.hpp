// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Contains helper functions for testing math functions

#pragma once

#include <random>

#include "tests/Unit/TestHelpers.hpp"

namespace test_math_helpers {
/// \ingroup TestHelpers
/// \brief Serializes and deserializes the MathFunction `function` and checks
/// that the resulting function gives the same results when applied to some
/// random input values
template <typename Function>
void test_pup_function(const Function& function) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> real_dis(-1, 1);
  const auto puped_function = serialize_and_deserialize(function);
  for (size_t i = 0; i < 100; ++i) {
    const double point = real_dis(gen);
    CAPTURE_PRECISE(point);
    CHECK(puped_function(point) == function(point));
    CHECK(puped_function.first_deriv(point) == function.first_deriv(point));
    CHECK(puped_function.second_deriv(point) == function.second_deriv(point));
  }
}

/// \cond
template <typename Function>
void test_pup_function(const std::unique_ptr<Function>& function) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> real_dis(-1, 1);
  const auto puped_function = serialize_and_deserialize(function);
  for (size_t i = 0; i < 100; ++i) {
    const double point = real_dis(gen);
    CAPTURE_PRECISE(point);
    CHECK(puped_function->operator()(point) == function->operator()(point));
    CHECK(puped_function->first_deriv(point) == function->first_deriv(point));
    CHECK(puped_function->second_deriv(point) == function->second_deriv(point));
  }
}
/// \endcond
}  // namespace test_math_helpers
