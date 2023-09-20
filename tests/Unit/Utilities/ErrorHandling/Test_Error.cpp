// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <exception>

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/Exceptions.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename ExceptionTypeToThrow>
void test() {
  CHECK_THROWS_MATCHES([]() { ERROR_AS("Test error", ExceptionTypeToThrow); }(),
                       ExceptionTypeToThrow,
                       Catch::Matchers::MessageMatches(
                           Catch::Matchers::ContainsSubstring("ERROR") &&
                           Catch::Matchers::ContainsSubstring("Test error")));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ErrorHandling.Error", "[Unit][ErrorHandling]") {
  CHECK_THROWS_MATCHES([]() { ERROR("Test error"); }(), SpectreError,
                       Catch::Matchers::MessageMatches(
                           Catch::Matchers::ContainsSubstring("ERROR") &&
                           Catch::Matchers::ContainsSubstring("Test error")));

  using exceptions_list =
      tmpl::list<std::logic_error, std::invalid_argument, std::domain_error,
                 std::length_error, std::out_of_range,

                 std::runtime_error, std::range_error, std::overflow_error,
                 std::underflow_error,

                 std::ios_base::failure,

                 SpectreError, SpectreAssert, convergence_error>;
  tmpl::for_each<exceptions_list>([](auto exception_type_v) {
    test<tmpl::type_from<decltype(exception_type_v)>>();
  });
}
