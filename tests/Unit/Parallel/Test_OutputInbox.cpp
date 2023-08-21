// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "Parallel/OutputInbox.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Parallel {
namespace {
struct TestInbox1 {
  using temporal_id = double;
  using type = std::map<temporal_id, std::pair<std::string, size_t>>;

  static std::string output_inbox(const type& inbox,
                                  const size_t padding_size) {
    std::stringstream ss{};
    const std::string pad(padding_size, ' ');

    ss << std::scientific << std::setprecision(10);
    ss << pad << "TestInbox1:\n";
    for (const auto& [time, data] : inbox) {
      ss << pad << " Time: " << time << ", data: " << data << "\n";
    }

    return ss.str();
  }
};

struct TestInbox2 {
  using temporal_id = std::string;
  using type = std::map<temporal_id, double>;

  static std::string output_inbox(const type& inbox,
                                  const size_t padding_size) {
    std::stringstream ss{};
    const std::string pad(padding_size, ' ');

    ss << std::scientific << std::setprecision(10);
    ss << pad << "TestInbox2:\n";
    for (const auto& [name, number] : inbox) {
      ss << pad << " Name: " << name << ", number: " << number << "\n";
    }

    return ss.str();
  }
};

SPECTRE_TEST_CASE("Unit.Parallel.OutputInbox", "[Unit][Parallel]") {
  tuples::TaggedTuple<TestInbox1, TestInbox2> inboxes{};
  std::map<double, std::pair<std::string, size_t>>& inbox_1 =
      tuples::get<TestInbox1>(inboxes);
  inbox_1[1.0] = std::make_pair("OneMississippi", 1_st);
  inbox_1[2.0] = std::make_pair("TwoMississippi", 2_st);
  std::map<std::string, double>& inbox_2 = tuples::get<TestInbox2>(inboxes);
  inbox_2["Dog"] = 5.5;
  inbox_2["Human"] = 0.6;
  inbox_2["Cat"] = 4.3;

  std::string result = output_inbox<TestInbox1>(inboxes, 0_st);
  std::string expected_result =
      "TestInbox1:\n"
      " Time: 1.0000000000e+00, data: (OneMississippi, 1)\n"
      " Time: 2.0000000000e+00, data: (TwoMississippi, 2)\n";

  CHECK(result == expected_result);

  result = output_inbox<TestInbox1>(inboxes, 1_st);
  expected_result =
      " TestInbox1:\n"
      "  Time: 1.0000000000e+00, data: (OneMississippi, 1)\n"
      "  Time: 2.0000000000e+00, data: (TwoMississippi, 2)\n";

  CHECK(result == expected_result);

  result = output_inbox<TestInbox2>(inboxes, 3_st);
  expected_result =
      "   TestInbox2:\n"
      "    Name: Cat, number: 4.3000000000e+00\n"
      "    Name: Dog, number: 5.5000000000e+00\n"
      "    Name: Human, number: 6.0000000000e-01\n";

  CHECK(result == expected_result);
}
}  // namespace
}  // namespace Parallel
