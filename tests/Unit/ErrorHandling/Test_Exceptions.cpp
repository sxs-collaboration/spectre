// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <stdexcept>
#include <string>

#include "ErrorHandling/Exceptions.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.ErrorHandling.Exceptions.convergence_error",
                  "[ErrorHandling][Unit]") {
  const std::string what = "Test throw";
  const auto thrower = [&what]() { throw convergence_error(what); };
  test_throw_exception(thrower, convergence_error(what));
  test_throw_exception(thrower, std::runtime_error(what));
}
