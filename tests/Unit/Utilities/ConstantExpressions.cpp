// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "Utilities/ConstantExpressions.hpp"

namespace {
template <typename T>
struct TestT {
  static constexpr T zero = 0;
  static constexpr T one = 1;
  static constexpr T two = 2;
  static constexpr T three = 3;
  static constexpr T four = 4;
  static constexpr T eight = 8;
  static constexpr T sixteen = 16;

  static_assert(one == two_to_the(zero),
                "Failed test Unit.Utilities.ConstantExpressions");
  static_assert(two == two_to_the(one),
                "Failed test Unit.Utilities.ConstantExpressions");
  static_assert(four == two_to_the(two),
                "Failed test Unit.Utilities.ConstantExpressions");
  static_assert(eight == two_to_the(three),
                "Failed test Unit.Utilities.ConstantExpressions");
  static_assert(sixteen == two_to_the(four),
                "Failed test Unit.Utilities.ConstantExpressions");
};

template struct TestT<int>;
template struct TestT<short>;
template struct TestT<long>;
template struct TestT<long long>;
template struct TestT<unsigned int>;
template struct TestT<unsigned short>;
template struct TestT<unsigned long>;
template struct TestT<unsigned long long>;
}  // namespace
