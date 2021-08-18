// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <map>
#include <string>
#include <unordered_map>

#include "Framework/TestHelpers.hpp"
#include "Time/TimeAndPrevious.hpp"
#include "Utilities/MakeString.hpp"

SPECTRE_TEST_CASE("Unit.Time.TimeAndPrevious", "[Unit][Time]") {
  const TimeAndPrevious time_and_previous{0.123, 0.1012};
  test_serialization(time_and_previous);
  const TimeAndPrevious same{0.123, 0.1012};

  CHECK(time_and_previous == same);
  CHECK_FALSE(time_and_previous != same);
  const TimeAndPrevious not_same{0.124, 0.101};
  CHECK(time_and_previous != not_same);
  CHECK_FALSE(time_and_previous == not_same);

  std::string string_of_object = MakeString{} << time_and_previous;
  std::string expected_string_of_object = MakeString{} << 0.123 << ", "
                                                       << 0.1012;
  CHECK(string_of_object == expected_string_of_object);

  TimeAndPreviousLessComparator comparator{};
  CHECK(comparator(time_and_previous, not_same));
  CHECK_FALSE(comparator(not_same, time_and_previous));
  CHECK_FALSE(comparator(time_and_previous, same));
  CHECK_FALSE(comparator(same, time_and_previous));
  CHECK(comparator(time_and_previous, 0.2));
  CHECK_FALSE(comparator(0.2, time_and_previous));
  CHECK(comparator(0.1, time_and_previous));
  CHECK_FALSE(comparator(time_and_previous, 0.1));

  // check that we can use TimeAndPrevious as a map key
  std::map<TimeAndPrevious, double, TimeAndPreviousLessComparator> test_map{};
  test_map.insert({TimeAndPrevious{0.123, 0.120}, 0.5});
  test_map.insert({TimeAndPrevious{0.120, 0.115}, 0.4});
  CHECK(test_map.begin()->first == TimeAndPrevious{0.120, 0.115});
  CHECK(test_map[TimeAndPrevious{0.123, 0.120}] == 0.5);

  // check that we can use TimeAndPrevious as an unordered_map key
  std::unordered_map<TimeAndPrevious, double> test_unordered_map{};
  test_unordered_map.insert({TimeAndPrevious{0.123, 0.120}, 0.5});
  test_unordered_map.insert({TimeAndPrevious{0.120, 0.115}, 0.4});
  CHECK(test_unordered_map[TimeAndPrevious{0.120, 0.115}] == 0.4);
  CHECK(test_map[TimeAndPrevious{0.123, 0.120}] == 0.5);
}
