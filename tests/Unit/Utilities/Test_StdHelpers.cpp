// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Framework/TestHelpers.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/StdHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.StdHelpers.Output", "[Utilities][Unit]") {
  NonStreamable ns{2};
  NonStreamable another_ns{4};

  std::list<int> my_list;
  CHECK(get_output(my_list) == "()");
  my_list = {1};
  CHECK(get_output(my_list) == "(1)");
  my_list = {{1, 2, 3, 4, 5}};
  CHECK(get_output(my_list) == "(1,2,3,4,5)");
  std::list<NonStreamable> ns_list{};
  CHECK(get_output(ns_list) == "()");
  ns_list = {ns};
  CHECK(get_output(ns_list) == "(UNSTREAMABLE)");
  ns_list = {{ns, ns}};
  CHECK(get_output(ns_list) == "(UNSTREAMABLE,UNSTREAMABLE)");

  std::vector<int> my_vector;
  CHECK(get_output(my_vector) == "()");
  my_vector = {1};
  CHECK(get_output(my_vector) == "(1)");
  my_vector = {{1, 2, 3, 4, 5}};
  CHECK(get_output(my_vector) == "(1,2,3,4,5)");
  std::vector<NonStreamable> ns_vector{};
  CHECK(get_output(ns_vector) == "()");
  ns_vector = {ns};
  CHECK(get_output(ns_vector) == "(UNSTREAMABLE)");
  ns_vector = {{ns, ns}};
  CHECK(get_output(ns_vector) == "(UNSTREAMABLE,UNSTREAMABLE)");

  std::deque<int> my_deque;
  CHECK(get_output(my_deque) == "()");
  my_deque = {1};
  CHECK(get_output(my_deque) == "(1)");
  my_deque = {{1, 2, 3, 4, 5}};
  CHECK(get_output(my_deque) == "(1,2,3,4,5)");
  std::deque<NonStreamable> ns_deque;
  CHECK(get_output(ns_deque) == "()");
  ns_deque = {ns};
  CHECK(get_output(ns_deque) == "(UNSTREAMABLE)");
  ns_deque = {{ns, ns}};
  CHECK(get_output(ns_deque) == "(UNSTREAMABLE,UNSTREAMABLE)");

  std::array<int, 0> a0{};
  CHECK(get_output(a0) == "()");
  std::array<int, 1> a1{{1}};
  CHECK(get_output(a1) == "(1)");
  std::array<int, 5> a5{{1, 2, 3, 4, 5}};
  CHECK(get_output(a5) == "(1,2,3,4,5)");
  std::array<NonStreamable, 0> ns_array0{};
  CHECK(get_output(ns_array0) == "()");
  std::array<NonStreamable, 1> ns_array1{{ns}};
  CHECK(get_output(ns_array1) == "(UNSTREAMABLE)");
  std::array<NonStreamable, 2> ns_array2 = {{ns, ns}};
  CHECK(get_output(ns_array2) == "(UNSTREAMABLE,UNSTREAMABLE)");

  auto tuple1 = std::make_tuple<int, double, std::string>(1, 1.87, "test");
  CHECK(get_output(tuple1) == "(1,1.87,test)");
  std::tuple<> tuple0{};
  CHECK(get_output(tuple0) == "()");
  auto tuple_ns = std::make_tuple<int, double, NonStreamable>(1, 1.87, {});
  CHECK(get_output(tuple_ns) == "(1,1.87,UNSTREAMABLE)");

  std::optional<int> opt{};
  CHECK(get_output(opt) == "--");
  opt = -42;
  CHECK(get_output(opt) == "-42");
  opt = std::nullopt;
  CHECK(get_output(opt) == "--");
  std::optional<NonStreamable> opt_ns{};
  CHECK(get_output(opt_ns) == "--");
  opt_ns = ns;
  CHECK(get_output(opt_ns) == "UNSTREAMABLE");
  opt_ns = std::nullopt;
  CHECK(get_output(opt_ns) == "--");

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
  std::unordered_map<int, NonStreamable> ns_value_unordered_map;
  CHECK(get_output(ns_value_unordered_map) == "()");
  CHECK(keys_of(ns_value_unordered_map) == "()");
  ns_value_unordered_map[1] = ns;
  CHECK(get_output(ns_value_unordered_map) == "([1,UNSTREAMABLE])");
  CHECK(keys_of(ns_value_unordered_map) == "(1)");
  ns_value_unordered_map[3] = ns;
  CHECK(get_output(ns_value_unordered_map) ==
        "([1,UNSTREAMABLE],[3,UNSTREAMABLE])");
  CHECK(keys_of(ns_value_unordered_map) == "(1,3)");
  std::unordered_map<NonStreamable, int> ns_key_unordered_map;
  CHECK(get_output(ns_key_unordered_map) == "()");
  CHECK(keys_of(ns_key_unordered_map) == "()");
  ns_key_unordered_map[ns] = 1;
  CHECK(get_output(ns_key_unordered_map) == "([UNSTREAMABLE,1])");
  CHECK(keys_of(ns_key_unordered_map) == "(UNSTREAMABLE)");
  ns_key_unordered_map[NonStreamable{4}] = 3;
  CHECK(get_output(ns_key_unordered_map) ==
        "([UNSTREAMABLE,1],[UNSTREAMABLE,3])");
  CHECK(keys_of(ns_key_unordered_map) == "(UNSTREAMABLE,UNSTREAMABLE)");

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

  std::map<int, NonStreamable> ns_value_map;
  CHECK(get_output(ns_value_map) == "()");
  CHECK(keys_of(ns_value_map) == "()");
  ns_value_map[1] = ns;
  CHECK(get_output(ns_value_map) == "([1,UNSTREAMABLE])");
  CHECK(keys_of(ns_value_map) == "(1)");
  ns_value_map[3] = ns;
  CHECK(get_output(ns_value_map) == "([1,UNSTREAMABLE],[3,UNSTREAMABLE])");
  CHECK(keys_of(ns_value_map) == "(1,3)");
  std::map<NonStreamable, int> ns_key_map;
  CHECK(get_output(ns_key_map) == "()");
  CHECK(keys_of(ns_key_map) == "()");
  ns_key_map[ns] = 1;
  CHECK(get_output(ns_key_map) == "([UNSTREAMABLE,1])");
  CHECK(keys_of(ns_key_map) == "(UNSTREAMABLE)");
  ns_key_map[NonStreamable{4}] = 3;
  CHECK(get_output(ns_key_map) == "([UNSTREAMABLE,1],[UNSTREAMABLE,3])");
  CHECK(keys_of(ns_key_map) == "(UNSTREAMABLE,UNSTREAMABLE)");

  std::unordered_set<int> my_unordered_set{1, 3, 4, 5};
  CHECK(get_output(my_unordered_set) == "(1,3,4,5)");
  std::unordered_set<int, boost::hash<int>> my_boost_unordered_set{1, 3, 4, 5};
  CHECK(get_output(my_boost_unordered_set) == "(1,3,4,5)");
  std::unordered_set<NonStreamable> ns_unordered_set{ns, another_ns};
  CHECK(get_output(ns_unordered_set) == "(UNSTREAMABLE,UNSTREAMABLE)");

  std::unordered_multiset<int> my_unordered_multiset{1, 3, 1, 5};
  CHECK(get_output(my_unordered_multiset) == "(1,1,3,5)");
  std::unordered_multiset<int, boost::hash<int>> my_boost_unordered_multiset{
      1, 3, 1, 5};
  CHECK(get_output(my_boost_unordered_multiset) == "(1,1,3,5)");
  std::unordered_multiset<NonStreamable> ns_unordered_multiset{ns, ns};
  CHECK(get_output(ns_unordered_multiset) == "(UNSTREAMABLE,UNSTREAMABLE)");

  std::set<int> my_set{1, 3, 4, 5};
  CHECK(get_output(my_set) == "(1,3,4,5)");
  std::unordered_set<NonStreamable> ns_set{ns, another_ns};
  CHECK(get_output(ns_set) == "(UNSTREAMABLE,UNSTREAMABLE)");

  auto my_unique = std::make_unique<double>(6.7);
  CHECK("6.7" == get_output(my_unique));
  auto my_shared = std::make_shared<double>(6.7);
  CHECK("6.7" == get_output(my_shared));

  auto my_pair = std::make_pair(7.8, "test"s);
  CHECK("(7.8, test)" == get_output(my_pair));
  auto ns_pair = std::make_pair(7.8, ns);
  CHECK("(7.8, UNSTREAMABLE)" == get_output(ns_pair));

  CHECK("1.19e+01 10 test" ==
        formatted_string("%1.2e %d %s", 11.87, 10, "test"));

  {
    std::ostringstream ss{};
    print_stl(ss, 5);
    CHECK(ss.str() == get_output(5));
  }
  {
    std::ostringstream ss{};
    print_stl(ss, std::vector{5});
    CHECK(ss.str() == get_output(std::vector{5}));
  }
}
