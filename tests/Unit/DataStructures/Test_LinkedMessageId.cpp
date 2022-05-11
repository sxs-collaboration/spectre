// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <map>
#include <string>
#include <unordered_map>

#include "DataStructures/LinkedMessageId.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeString.hpp"

namespace {
void test_id() {
  const LinkedMessageId<int> id_one_nothing{1, {}};
  const LinkedMessageId<int> id_one_two{1, 2};
  const LinkedMessageId<int> id_two_nothing{2, {}};
  CHECK(id_one_nothing == id_one_nothing);
  CHECK_FALSE(id_one_nothing != id_one_nothing);
  CHECK_FALSE(id_one_nothing == id_one_two);
  CHECK(id_one_nothing != id_one_two);
  CHECK_FALSE(id_one_nothing == id_two_nothing);
  CHECK(id_one_nothing != id_two_nothing);
  CHECK(get_output(id_one_two) == "1 (2)");

  const LinkedMessageId<double> time_and_previous{0.123, 0.1012};
  test_serialization(time_and_previous);
  const LinkedMessageId<double> same{0.123, 0.1012};

  CHECK(time_and_previous == same);
  CHECK_FALSE(time_and_previous != same);
  const LinkedMessageId<double> not_same{0.124, 0.101};
  CHECK(time_and_previous != not_same);
  CHECK_FALSE(time_and_previous == not_same);

  std::string string_of_object = MakeString{} << time_and_previous;
  std::string expected_string_of_object = MakeString{} << 0.123 << " ("
                                                       << 0.1012 << ")";
  CHECK(string_of_object == expected_string_of_object);

  LinkedMessageIdLessComparator<double> comparator{};
  CHECK(comparator(time_and_previous, not_same));
  CHECK_FALSE(comparator(not_same, time_and_previous));
  CHECK_FALSE(comparator(time_and_previous, same));
  CHECK_FALSE(comparator(same, time_and_previous));
  CHECK(comparator(time_and_previous, 0.2));
  CHECK_FALSE(comparator(0.2, time_and_previous));
  CHECK(comparator(0.1, time_and_previous));
  CHECK_FALSE(comparator(time_and_previous, 0.1));

  // check that we can use LinkedMessageId<double> as a map key
  std::map<LinkedMessageId<double>, double,
           LinkedMessageIdLessComparator<double>>
      test_map{};
  test_map.insert({LinkedMessageId<double>{0.123, 0.120}, 0.5});
  test_map.insert({LinkedMessageId<double>{0.120, 0.115}, 0.4});
  CHECK(test_map.begin()->first == LinkedMessageId<double>{0.120, 0.115});
  CHECK(test_map[LinkedMessageId<double>{0.123, 0.120}] == 0.5);

  // check that we can use LinkedMessageId<double> as an unordered_map key
  std::unordered_map<LinkedMessageId<double>, double> test_unordered_map{};
  test_unordered_map.insert({LinkedMessageId<double>{0.123, 0.120}, 0.5});
  test_unordered_map.insert({LinkedMessageId<double>{0.120, 0.115}, 0.4});
  CHECK(test_unordered_map[LinkedMessageId<double>{0.120, 0.115}] == 0.4);
  CHECK(test_map[LinkedMessageId<double>{0.123, 0.120}] == 0.5);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.LinkedMessageId",
                  "[Unit][DataStructures]") {
  test_id();
}
