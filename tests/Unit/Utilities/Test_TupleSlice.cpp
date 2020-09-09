// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <tuple>

#include "Framework/TestHelpers.hpp"
#include "Utilities/TupleSlice.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.TupleSlice", "[Utilities][Unit]") {
  CHECK(tuple_slice<1, 3>(std::tuple<int, float, double, std::string>{
            1, 2., 3., ""}) == std::tuple<float, double>{2., 3.});
  CHECK(tuple_head<2>(std::tuple<int, float, double, std::string>{
            1, 2., 3., ""}) == std::tuple<int, float>{1, 2.});
  CHECK(tuple_tail<2>(std::tuple<int, float, double, std::string>{
            1, 2., 3., ""}) == std::tuple<double, std::string>{3., ""});
  CHECK(tuple_slice<0, 1>(std::tuple<int>{1}) == std::tuple<int>{1});
  CHECK(tuple_head<1>(std::tuple<int>{1}) == std::tuple<int>{1});
  CHECK(tuple_tail<1>(std::tuple<int>{1}) == std::tuple<int>{1});
  CHECK(tuple_slice<0, 0>(std::tuple<int>{1}) == std::tuple<>{});
  CHECK(tuple_head<0>(std::tuple<int>{1}) == std::tuple<>{});
  CHECK(tuple_tail<0>(std::tuple<int>{1}) == std::tuple<>{});
  CHECK(tuple_slice<0, 0>(std::tuple<>{}) == std::tuple<>{});
  CHECK(tuple_head<0>(std::tuple<>{}) == std::tuple<>{});
  CHECK(tuple_tail<0>(std::tuple<>{}) == std::tuple<>{});
  {
    INFO("Copy tuple of reference types");
    ConstructionObserver a{};
    ConstructionObserver& b = a;
    const std::tuple<ConstructionObserver, ConstructionObserver&> tuple{a, b};
    // Assert that tuple_slice behaves like std::tuple_cat
    static_assert(std::is_same_v<decltype(tuple_slice<0, 2>(tuple)),
                                 decltype(std::tuple_cat(tuple))>);
    const auto sliced_tuple = tuple_slice<0, 2>(tuple);
    CHECK(b.status == "initial");
    CHECK(std::get<0>(sliced_tuple).status == "copy-constructed");
    CHECK(std::get<1>(sliced_tuple).status == "initial");
  }
  {
    INFO("Move tuple of reference types");
    ConstructionObserver a{};
    ConstructionObserver& b = a;
    ConstructionObserver c{};
    const ConstructionObserver& d = a;
    ConstructionObserver e{};
    std::tuple<ConstructionObserver, ConstructionObserver&,
               ConstructionObserver&&, const ConstructionObserver&,
               const ConstructionObserver&&>
        tuple{a, b, std::move(c), d, std::move(e)};
    // Assert that tuple_slice behaves like std::tuple_cat
    static_assert(std::is_same_v<decltype(tuple_slice<0, 5>(std::move(tuple))),
                                 decltype(std::tuple_cat(std::move(tuple)))>);
    const auto moved_tuple = tuple_slice<0, 5>(std::move(tuple));
    CHECK(b.status == "initial");
    CHECK(std::get<0>(moved_tuple).status == "move-constructed");
    CHECK(std::get<1>(moved_tuple).status == "initial");
    CHECK(std::get<2>(moved_tuple).status == "initial");
    CHECK(std::get<3>(moved_tuple).status == "initial");
    CHECK(std::get<4>(moved_tuple).status == "initial");
  }
  {
    INFO("Works with non-tuple containers");
    CHECK(tuple_head<1>(std::array<double, 2>{{1., 2.}}) ==
          std::tuple<double>{{1.}});
    CHECK(tuple_head<1>(std::pair<int, float>{1, 2.}) == std::tuple<int>{1});
  }
}
