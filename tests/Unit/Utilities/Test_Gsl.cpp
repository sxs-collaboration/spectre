// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <numeric>
#include <vector>

#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <typename T>
void func(const gsl::not_null<T*> t) { *t += 2; }

void test_make_not_null() {
  int x = 5;
  func(make_not_null(&x));
  CHECK(x == 7);
}

void test_span_stream() {
  std::vector<double> a(4);
  std::iota(a.begin(), a.end(), 1.3);
  gsl::span<const double> span_a(a.data(), a.size());
  CHECK(get_output(span_a) == "(1.3,2.3,3.3,4.3)");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.Gsl", "[Unit][Utilities]") {
  test_make_not_null();
  test_span_stream();
}
