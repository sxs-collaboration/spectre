// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Options/Options.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"

namespace Parallel {
namespace Tags {
struct IntegerList : db::SimpleTag {
  using type = std::array<int, 3>;
};

struct UniquePtrIntegerList : db::BaseTag {
  using type = std::unique_ptr<std::array<int, 3>>;
};
}  // namespace Tags

namespace {
struct Metavars {
  using const_global_cache_tags =
      tmpl::list<Tags::IntegerList, Tags::UniquePtrIntegerList>;
  using component_list = tmpl::list<>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.ConstGlobalCacheDataBox", "[Unit][Parallel]") {
  tuples::TaggedTuple<Tags::IntegerList, Tags::UniquePtrIntegerList> tuple{};
  tuples::get<Tags::IntegerList>(tuple) = std::array<int, 3>{{-1, 3, 7}};
  tuples::get<Tags::UniquePtrIntegerList>(tuple) =
      std::make_unique<std::array<int, 3>>(std::array<int, 3>{{1, 5, -8}});
  ConstGlobalCache<Metavars> cache{std::move(tuple)};
  auto box =
      db::create<db::AddSimpleTags<Tags::ConstGlobalCacheImpl<Metavars>>,
                 db::AddComputeTags<
                     Tags::FromConstGlobalCache<Tags::IntegerList>,
                     Tags::FromConstGlobalCache<Tags::UniquePtrIntegerList>>>(
          &cpp17::as_const(cache));
  CHECK(db::get<Tags::ConstGlobalCache>(box) == &cache);
  CHECK(std::array<int, 3>{{-1, 3, 7}} == db::get<Tags::IntegerList>(box));
  CHECK(std::array<int, 3>{{1, 5, -8}} ==
        db::get<Tags::UniquePtrIntegerList>(box));
  CHECK(&Parallel::get<Tags::IntegerList>(cache) ==
        &db::get<Tags::IntegerList>(box));
  CHECK(&Parallel::get<Tags::UniquePtrIntegerList>(cache) ==
        &db::get<Tags::UniquePtrIntegerList>(box));

  tuples::TaggedTuple<Tags::IntegerList, Tags::UniquePtrIntegerList> tuple2{};
  tuples::get<Tags::IntegerList>(tuple2) = std::array<int, 3>{{10, -3, 700}};
  tuples::get<Tags::UniquePtrIntegerList>(tuple2) =
      std::make_unique<std::array<int, 3>>(std::array<int, 3>{{100, -7, -300}});
  ConstGlobalCache<Metavars> cache2{std::move(tuple2)};
  db::mutate<Tags::ConstGlobalCache>(
      make_not_null(&box),
      [&cache2](
          const gsl::not_null<const Parallel::ConstGlobalCache<Metavars>**> t) {
        *t = std::addressof(cache2);
      });

  CHECK(db::get<Tags::ConstGlobalCache>(box) == &cache2);
  CHECK(std::array<int, 3>{{10, -3, 700}} == db::get<Tags::IntegerList>(box));
  CHECK(std::array<int, 3>{{100, -7, -300}} ==
        db::get<Tags::UniquePtrIntegerList>(box));
  CHECK(&Parallel::get<Tags::IntegerList>(cache2) ==
        &db::get<Tags::IntegerList>(box));
  CHECK(&Parallel::get<Tags::UniquePtrIntegerList>(cache2) ==
        &db::get<Tags::UniquePtrIntegerList>(box));

  TestHelpers::db::test_base_tag<Tags::ConstGlobalCache>("ConstGlobalCache");
  TestHelpers::db::test_simple_tag<Tags::ConstGlobalCacheImpl<Metavars>>(
      "ConstGlobalCache");
  TestHelpers::db::test_compute_tag<
      Tags::FromConstGlobalCache<Tags::IntegerList>>(
      "FromConstGlobalCache(IntegerList)");
  TestHelpers::db::test_compute_tag<
      Tags::FromConstGlobalCache<Tags::UniquePtrIntegerList>>(
      "FromConstGlobalCache(UniquePtrIntegerList)");
}
}  // namespace Parallel
