// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Utilities/GetOutput.hpp"
#include "Utilities/StdHelpers.hpp"

// IWYU pragma: no_forward_declare boost::hash

SPECTRE_TEST_CASE("Unit.Utilities.StdHelpers.Output", "[Utilities][Unit]") {
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

  std::deque<int> my_deque;
  CHECK(get_output(my_deque) == "()");
  my_deque = {1};
  CHECK(get_output(my_deque) == "(1)");
  my_deque = {{1, 2, 3, 4, 5}};
  CHECK(get_output(my_deque) == "(1,2,3,4,5)");

  std::array<int, 0> a0{};
  CHECK(get_output(a0) == "()");
  std::array<int, 1> a1{{1}};
  CHECK(get_output(a1) == "(1)");
  std::array<int, 5> a5{{1, 2, 3, 4, 5}};
  CHECK(get_output(a5) == "(1,2,3,4,5)");

  auto tuple1 = std::make_tuple<int, double, std::string>(1, 1.87, "test");
  CHECK(get_output(tuple1) == "(1,1.87,test)");
  std::tuple<> tuple0{};
  CHECK(get_output(tuple0) == "()");

  std::unordered_map<std::string, int, boost::hash<std::string>>
      my_unordered_map;
  CHECK(get_output(my_unordered_map) == "()");
  CHECK(keys_of(my_unordered_map) == "()");
  my_unordered_map["aaa"] = 1;
  CHECK(get_output(my_unordered_map) == "([aaa,1])");
  CHECK(keys_of(my_unordered_map) == "(aaa)");
  my_unordered_map["bbb"] = 2;
  my_unordered_map["ccc"] = 3;
  my_unordered_map["ddd"] = 4;
  my_unordered_map["eee"] = 5;
  CHECK(get_output(my_unordered_map) ==
        "([aaa,1],[bbb,2],[ccc,3],[ddd,4],[eee,5])");
  CHECK(keys_of(my_unordered_map) == "(aaa,bbb,ccc,ddd,eee)");

  // check map with some other comparison op
  std::map<std::string, int, std::greater<>> my_map;
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
  CHECK(get_output(my_unordered_set) == "(1,3,4,5)");

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
