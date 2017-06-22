// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <list>
#include <map>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

TEST_CASE("Unit.Utilities.StdHelpers.Output", "[Utilities][Unit]") {
  // We don't test unordered containers because the order of the output depends
  // on not just libc++ vs. stdlibc++ but also the OS, etc.
  std::list<int> my_list;
  CHECK(get_output(my_list) == "()");
  my_list = {1};
  CHECK(get_output(my_list) == "(1)");
  my_list = {{1, 2, 3, 4, 5}};
  CHECK(get_output(my_list) == "(1,2,3,4,5)");

  std::vector<int> my_vector;
  CHECK(get_output(my_vector) == "()");
  my_vector = {1};
  CHECK(get_output(my_vector) == "(1)");
  my_vector = {{1, 2, 3, 4, 5}};
  CHECK(get_output(my_vector) == "(1,2,3,4,5)");

  std::array<int, 0> a0{};
  CHECK(get_output(a0) == "()");
  std::array<int, 1> a1{{1}};
  CHECK(get_output(a1) == "(1)");
  std::array<int, 5> a5{{1, 2, 3, 4, 5}};
  CHECK(get_output(a5) == "(1,2,3,4,5)");

  auto tuple1 = std::make_tuple<int, double, std::string>(1, 1.87, "test");
  CHECK(get_output(tuple1) == "(1, 1.87, test)");
  std::tuple<> tuple0{};
  CHECK(get_output(tuple0) == "()");

  // check map with some other comparison op
  std::map<std::string, int, std::greater<std::string>> my_map;  // NOLINT
  CHECK(get_output(my_map) == "()");
  CHECK(keys_of(my_map) == "()");
  my_map["aaa"] = 1;
  CHECK(get_output(my_map) == "([aaa,1])");
  CHECK(keys_of(my_map) == "(aaa)");
  my_map["bbb"] = 2;
  my_map["ccc"] = 3;
  my_map["ddd"] = 4;
  my_map["eee"] = 5;
  CHECK(get_output(my_map) == "([eee,5],[ddd,4],[ccc,3],[bbb,2],[aaa,1])");
  CHECK(keys_of(my_map) == "(eee,ddd,ccc,bbb,aaa)");

  std::unordered_set<int> my_unordered_set{1, 3, 4, 5};
  std::string my_unordered_set_output = get_output(my_unordered_set);
  CHECK(my_unordered_set_output.find('1') != std::string::npos);
  CHECK(my_unordered_set_output.find('3') != std::string::npos);
  CHECK(my_unordered_set_output.find('4') != std::string::npos);
  CHECK(my_unordered_set_output.find('5') != std::string::npos);
  CHECK(my_unordered_set_output.find('2') == std::string::npos);
  CHECK(my_unordered_set_output.find('0') == std::string::npos);

  std::set<int> my_set{1, 3, 4, 5};
  CHECK(get_output(my_set) == "(1,3,4,5)");

  auto my_unique = std::make_unique<double>(6.7);
  CHECK("6.7" == get_output(my_unique));
  auto my_shared = std::make_shared<double>(6.7);
  CHECK("6.7" == get_output(my_shared));

  auto my_pair = std::make_pair(7.8, "test"s);
  CHECK("(7.8, test)" == get_output(my_pair));

  CHECK("1.19e+01 10 test" ==
        formatted_string("%1.2e %d %s", 11.87, 10, "test"));
}
