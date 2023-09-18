// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <stdexcept>
#include <string>

#include "Framework/TestHelpers.hpp"
#include "Utilities/ErrorHandling/Exceptions.hpp"

SPECTRE_TEST_CASE("Unit.ErrorHandling.Exceptions.convergence_error",
                  "[ErrorHandling][Unit]") {
  const std::string what = "Test throw";
  const auto thrower = [&what]() { throw convergence_error(what); };
  test_throw_exception(thrower, convergence_error(what));
  test_throw_exception(thrower, std::runtime_error(what));
}
