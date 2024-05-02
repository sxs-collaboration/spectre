// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>

#include "Framework/TestHelpers.hpp"
#include "Time/TimeStepRequest.hpp"

namespace {
void check_eq(std::optional<double> TimeStepRequest::* const field) {
  const TimeStepRequest request{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
  auto request2 = request;
  (request2.*field).reset();
  CHECK(request != request2);
  CHECK_FALSE(request == request2);
}

SPECTRE_TEST_CASE("Unit.Time.TimeStepRequest", "[Unit][Time]") {
  CHECK(TimeStepRequest{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}} ==
        TimeStepRequest{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}});
  CHECK_FALSE(TimeStepRequest{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}} !=
              TimeStepRequest{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}});

  check_eq(&TimeStepRequest::size_goal);
  check_eq(&TimeStepRequest::size);
  check_eq(&TimeStepRequest::end);
  check_eq(&TimeStepRequest::size_hard_limit);
  check_eq(&TimeStepRequest::end_hard_limit);

  test_serialization(TimeStepRequest{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}});
}
}  // namespace
