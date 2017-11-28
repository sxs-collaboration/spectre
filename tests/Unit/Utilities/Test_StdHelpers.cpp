// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/functional/hash.hpp>
#include <catch.hpp>
#include <list>
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
// We wish to explicitly test implicit type conversion when adding std::arrays
// of different fundamentals, so we supress this warning.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include "Utilities/StdHelpers.hpp"
#pragma GCC diagnostic pop
#include "tests/Unit/TestHelpers.hpp"

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

SPECTRE_TEST_CASE("Unit.Utilities.StdHelpers.StdArrayArithmetic",
                  "[DataStructures][Unit]") {
  const size_t Dim = 3;

  std::array<double, Dim> p1{{2.3, -1.4, 0.2}};
  std::array<double, Dim> p2{{-12.4, 4.5, 2.6}};

  const std::array<double, Dim> expected_plus{{-10.1, 3.1, 2.8}};
  const std::array<double, Dim> expected_minus{{14.7, -5.9, -2.4}};

  const auto plus = p1 + p2;
  const auto minus = p1 - p2;

  for (size_t i = 0; i < Dim; ++i) {
    CHECK(gsl::at(plus, i) == approx(gsl::at(expected_plus, i)));
    CHECK(gsl::at(minus, i) == approx(gsl::at(expected_minus, i)));
  }

  p1 += expected_plus;
  p2 -= expected_minus;

  const std::array<double, Dim> expected_plus_equal{{-7.8, 1.7, 3.}};
  const std::array<double, Dim> expected_minus_equal{{-27.1, 10.4, 5.}};

  for (size_t i = 0; i < Dim; ++i) {
    CHECK(gsl::at(p1, i) == approx(gsl::at(expected_plus_equal, i)));
    CHECK(gsl::at(p2, i) == approx(gsl::at(expected_minus_equal, i)));
  }

  const double scale = -1.8;
  const auto left_scaled_array = scale * p1;
  const std::array<double, Dim> expected_left_scaled_array{
      {14.04, -3.06, -5.4}};
  const auto right_scaled_array = p1 * scale;
  const auto array_divided_by_double = p1 / (1 / scale);

  for (size_t i = 0; i < Dim; ++i) {
    CHECK(gsl::at(left_scaled_array, i) ==
          approx(gsl::at(expected_left_scaled_array, i)));
    CHECK(gsl::at(left_scaled_array, i) ==
          approx(gsl::at(right_scaled_array, i)));
    CHECK(gsl::at(left_scaled_array, i) ==
          approx(gsl::at(array_divided_by_double, i)));
  }

  const auto neg_p1 = -p1;
  const auto expected_neg_p1 = -1. * p1;
  for (size_t i = 0; i < Dim; ++i) {
    CHECK(gsl::at(neg_p1, i) == approx(gsl::at(expected_neg_p1, i)));
  }

  const std::array<double, 2> double_array{{2.2, -1.0}};
  const std::array<int, 2> int_array{{1, 3}};
  const std::array<size_t, 2> size_array{{2, 10}};
  const std::array<double, 2> double_plus_int{{3.2, 2.0}};
  const std::array<double, 2> double_plus_size{{4.2, 9.0}};
  const std::array<size_t, 2> int_plus_size{{3, 13}};
  const std::array<size_t, 2> size_minus_int{{1, 7}};
  CHECK(double_array + int_array == double_plus_int);
  CHECK(double_array + size_array == double_plus_size);
  CHECK(int_array + size_array == int_plus_size);
  CHECK(int_array + double_array == double_plus_int);
  CHECK(size_array + double_array == double_plus_size);
  CHECK(size_array + int_array == int_plus_size);
  CHECK(size_array - -int_array == int_plus_size);
  CHECK(size_array - -double_array == double_plus_size);
  CHECK(size_array - int_array == size_minus_int);
}

SPECTRE_TEST_CASE("Unit.Utilities.StdHelpers.StdArrayMagnitude",
                  "[DataStructures][Unit]") {
  std::array<double, 1> p1{{-2.5}};
  CHECK(2.5 == magnitude(p1));
  const std::array<double, 2> p2{{3., -4.}};
  CHECK(magnitude(p2) == approx(5.));
  const std::array<double, 3> p3{{-2., 10., 11.}};
  CHECK(magnitude(p3) == approx(15.));
}

SPECTRE_TEST_CASE("Unit.Utilities.StdHelpers.AllButSpecifiedElementOf",
                  "[DataStructures][Unit]") {
  const std::array<size_t, 3> a3{{5, 2, 3}};
  const std::array<size_t, 2> a2{{2, 3}};
  const std::array<size_t, 1> a1{{3}};
  const std::array<size_t, 0> a0{{}};
  CHECK(a2 == all_but_specified_element_of<0>(a3));
  CHECK(a1 == all_but_specified_element_of<0>(a2));
  CHECK(a0 == all_but_specified_element_of<0>(a1));
  const std::array<size_t, 2> b2{{5, 3}};
  const std::array<size_t, 1> b1{{5}};
  auto c2 = all_but_specified_element_of<1>(a3);
  CHECK(b2 == c2);
  auto c1 = all_but_specified_element_of<1>(b2);
  CHECK(b1 == c1);
  CHECK(a0 == all_but_specified_element_of<0>(b1));
}
