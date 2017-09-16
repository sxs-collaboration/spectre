// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/Gsl.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
template <typename T>
void func(const gsl::not_null<T*> t) { *t += 2; }
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.Gsl.make_not_null", "[Unit][Utilities]") {
  int x = 5;
  func(make_not_null(&x));
  CHECK(x == 7);
}
