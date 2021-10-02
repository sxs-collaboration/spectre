// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Elliptic/Utilities/ApplyAt.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace elliptic::util {

namespace {

// [apply_at_tags]
struct MapTag : db::SimpleTag {
  using type = std::map<int, std::string>;
};
struct NonMapTag : db::SimpleTag {
  using type = int;
};
struct NestedMapTag : db::SimpleTag {
  using type = std::map<int, std::unordered_map<std::string, bool>>;
};
struct DirectionMapTag : db::SimpleTag {
  using type = DirectionMap<1, bool>;
};
// [apply_at_tags]

void test_apply_at() {
  const auto check = [](const auto& box) {
    // [apply_at_example]
    apply_at<tmpl::list<MapTag, NonMapTag, NestedMapTag>,
             tmpl::list<NonMapTag>>(
        [](const std::string& arg1, const int arg0,
           const std::unordered_map<std::string, bool>& arg2) {
          CHECK(arg0 == 1);
          CHECK(arg1 == "A");
          CHECK(arg2.at("key") == true);
        },
        box, 0);
    // [apply_at_example]
    apply_at<tmpl::list<NestedMapTag, NonMapTag>, tmpl::list<NonMapTag>>(
        [](const bool arg0, const int arg1) {
          CHECK(arg0 == true);
          CHECK(arg1 == 1);
        },
        box, std::make_tuple(0, "key"));
    apply_at<tmpl::list<DirectionMapTag>, tmpl::list<>>(
        [](const bool arg0) { CHECK(arg0); }, box, Direction<1>::lower_xi());
  };
  check(db::create<
        db::AddSimpleTags<MapTag, NonMapTag, NestedMapTag, DirectionMapTag>>(
      std::map<int, std::string>{{0, "A"}}, 1,
      std::map<int, std::unordered_map<std::string, bool>>{
          {0, {{"key", true}}}},
      DirectionMap<1, bool>{{Direction<1>::lower_xi(), true}}));
  check(tuples::TaggedTuple<MapTag, NonMapTag, NestedMapTag, DirectionMapTag>(
      std::map<int, std::string>{{0, "A"}}, 1,
      std::map<int, std::unordered_map<std::string, bool>>{
          {0, {{"key", true}}}},
      DirectionMap<1, bool>{{Direction<1>::lower_xi(), true}}));
}

void test_mutate_apply_at() {
  const auto check = [](auto box) {
    // [mutate_apply_at_example]
    mutate_apply_at<tmpl::list<MapTag>, tmpl::list<NonMapTag, NestedMapTag>,
                    tmpl::list<NonMapTag>>(
        [](const gsl::not_null<std::string*> mutate_arg, const int arg0,
           const std::unordered_map<std::string, bool>& arg2) {
          *mutate_arg = "B";
          CHECK(arg0 == 1);
          CHECK(arg2.at("key") == true);
        },
        make_not_null(&box), 0);
    // [mutate_apply_at_example]
    CHECK(db::get<MapTag>(box) == std::map<int, std::string>{{0, "B"}});
    mutate_apply_at<tmpl::list<NestedMapTag, NonMapTag>, tmpl::list<>,
                    tmpl::list<NonMapTag>>(
        [](const gsl::not_null<bool*> arg0, const gsl::not_null<int*> arg1) {
          CHECK(*arg0 == true);
          *arg0 = false;
          CHECK(*arg1 == 1);
          *arg1 = 2;
        },
        make_not_null(&box), std::make_tuple(0, "key"));
    CHECK(db::get<NestedMapTag>(box) ==
          std::map<int, std::unordered_map<std::string, bool>>{
              {0, {{"key", false}}}});
    CHECK(db::get<NonMapTag>(box) == 2);
    mutate_apply_at<tmpl::list<DirectionMapTag>, tmpl::list<>, tmpl::list<>>(
        [](const gsl::not_null<bool*> mutate_arg) {
          CHECK(*mutate_arg);
          *mutate_arg = false;
        },
        make_not_null(&box), Direction<1>::lower_xi());
    CHECK(db::get<DirectionMapTag>(box) ==
          DirectionMap<1, bool>{{Direction<1>::lower_xi(), false}});
  };
  check(db::create<
        db::AddSimpleTags<MapTag, NonMapTag, NestedMapTag, DirectionMapTag>>(
      std::map<int, std::string>{}, 1,
      std::map<int, std::unordered_map<std::string, bool>>{
          {0, {{"key", true}}}},
      DirectionMap<1, bool>{{Direction<1>::lower_xi(), true}}));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.ApplyAt",
                  "[Unit][DataStructures]") {
  test_apply_at();
  test_mutate_apply_at();
}

}  // namespace elliptic::util
