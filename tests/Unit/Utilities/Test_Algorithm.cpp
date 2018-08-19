// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <deque>
#include <set>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Utilities/Algorithm.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.indices_of", "[Unit][Utilities]") {
  const auto helper = [](const auto& data) {
    CHECK(std::set<size_t>{0, 5, 3} == [&data]() {
      const auto t = indices_of<3>(
          data, [](const double lhs, const double rhs) { return lhs < rhs; });
      return std::set<size_t>(t.begin(), t.end());
    }());
    CHECK(std::array<size_t, 3>{{3, 0, 5}} ==
          sorted_indices_of<3>(data, [](const double lhs, const double rhs) {
            return lhs < rhs;
          }));

    CHECK(std::set<size_t>{0, 6, 5} == [&data]() {
      const auto t =
          indices_of<3>(data, [](const double lhs, const double rhs) {
            return std::abs(lhs) < std::abs(rhs);
          });
      return std::set<size_t>(t.begin(), t.end());
    }());
    CHECK(std::array<size_t, 3>{{0, 5, 6}} ==
          sorted_indices_of<3>(data, [](const double lhs, const double rhs) {
            return std::abs(lhs) < std::abs(rhs);
          }));

    CHECK(std::set<size_t>{4, 1, 2} == [&data]() {
      const auto t = indices_of<3>(
          data, [](const double lhs, const double rhs) { return lhs > rhs; });
      return std::set<size_t>(t.begin(), t.end());
    }());
    CHECK(std::array<size_t, 3>{{4, 2, 1}} ==
          sorted_indices_of<3>(data, [](const double lhs, const double rhs) {
            return lhs > rhs;
          }));

    CHECK(std::set<size_t>{3, 4, 2} == [&data]() {
      const auto t =
          indices_of<3>(data, [](const double lhs, const double rhs) {
            return std::abs(lhs) > std::abs(rhs);
          });
      return std::set<size_t>(t.begin(), t.end());
    }());
    CHECK(std::array<size_t, 3>{{3, 4, 2}} ==
          sorted_indices_of<3>(data, [](const double lhs, const double rhs) {
            return std::abs(lhs) > std::abs(rhs);
          }));
  };
  helper(std::array<double, 7>{{-1.0, 7.8, 9.3, -22.8, 9.5, 4.3, 5.2}});
  helper(std::deque<double>{-1.0, 7.8, 9.3, -22.8, 9.5, 4.3, 5.2});
  helper(std::vector<double>{-1.0, 7.8, 9.3, -22.8, 9.5, 4.3, 5.2});
  helper(DataVector{-1.0, 7.8, 9.3, -22.8, 9.5, 4.3, 5.2});
}
