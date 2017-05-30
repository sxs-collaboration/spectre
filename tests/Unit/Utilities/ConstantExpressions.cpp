// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "Utilities/ConstantExpressions.hpp"

namespace {
template <typename T>
void TestT() {
  const T zero = 0;
  const T one = 1;
  const T two = 2;
  const T three = 3;
  const T four = 4;
  const T eight = 8;
  const T sixteen = 16;

  CHECK((one == two_to_the(zero)));

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
}
}  // namespace

TEST_CASE("Unit.Utilities.ConstantExpressions", "[Utilities][Unit]") {
  TestT<int>();
  TestT<short>();
  TestT<long>();
  TestT<long long>();
  TestT<unsigned int>();
  TestT<unsigned short>();
  TestT<unsigned long>();
  TestT<unsigned long long>();
  TestT<std::size_t>();
}
