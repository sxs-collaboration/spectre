// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "Utilities/Deferred.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
/// [functions_used]
struct func {
  double operator()() const { return 8.2; }
};

double dummy() { return 6.7; }

struct func2 {
  double operator()(const double& t) const { return t; }
};

double lazy_function(const double t) { return 10.0 * t; }
/// [functions_used]

void simple_deferred() {
  /// [deferred_with_update]
  auto obj = Deferred<double>(3.8);
  CHECK(3.8 == obj.get());
  auto& obj_val = obj.mutate();
  CHECK(3.8 == obj_val);
  obj_val = 5.0;
  CHECK(5.0 == obj.get());
  /// [deferred_with_update]
}

void single_call_deferred() {
  /// [make_deferred_with_function_object]
  auto def = make_deferred(func{});
  CHECK(8.2 == def.get());
  /// [make_deferred_with_function_object]

  /// [make_deferred_with_function]
  auto def2 = make_deferred(dummy);
  CHECK(6.7 == def2.get());
  /// [make_deferred_with_function]

  const auto function_name = dummy;
  auto def3 = make_deferred(function_name);
  CHECK(6.7 == def3.get());
}

void deferred_as_argument_to_deferred() {
  /// [make_deferred_with_deferred_arg]
  auto def2 = make_deferred(func2{}, 6.82);
  auto def3 = make_deferred(lazy_function, def2);
  CHECK(68.2 == def3.get());
  CHECK(6.82 == def2.get());
  /// [make_deferred_with_deferred_arg]
}
}  // namespace

TEST_CASE("Unit.Utilities.Deferred", "[Utilities][Unit]") {
  simple_deferred();
  single_call_deferred();
  deferred_as_argument_to_deferred();
}

// [[OutputRegex, Cannot mutate a computed Deferred]]
[[noreturn]] TEST_CASE("Unit.Utilities.Deferred.FailAlter",
                       "[Utilities][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto def = make_deferred(func{});
  auto& mutate = def.mutate();
  ERROR("Bad test end");
#endif
}
