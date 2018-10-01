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

namespace Parallel {
namespace OptionTags {
struct IntegerList : db::BaseTag {
  using type = std::array<int, 3>;
  static constexpr OptionString help = {"Help"};
};

struct UniquePtrIntegerList : db::BaseTag {
  using type = std::unique_ptr<std::array<int, 3>>;
  static constexpr OptionString help = {"Help"};
};
}  // namespace OptionTags

namespace {
struct Metavars {
  using const_global_cache_tag_list =
      tmpl::list<OptionTags::IntegerList, OptionTags::UniquePtrIntegerList>;
  using component_list = tmpl::list<>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.ConstGlobalCacheDataBox", "[Unit][Parallel]") {
  tuples::TaggedTuple<OptionTags::IntegerList, OptionTags::UniquePtrIntegerList>
      tuple{};
  tuples::get<OptionTags::IntegerList>(tuple) = std::array<int, 3>{{-1, 3, 7}};
  tuples::get<OptionTags::UniquePtrIntegerList>(tuple) =
      std::make_unique<std::array<int, 3>>(std::array<int, 3>{{1, 5, -8}});
  ConstGlobalCache<Metavars> cache{std::move(tuple)};
  auto box = db::create<
      db::AddSimpleTags<Tags::ConstGlobalCacheImpl<Metavars>>,
      db::AddComputeTags<
          Tags::FromConstGlobalCache<OptionTags::IntegerList>,
          Tags::FromConstGlobalCache<OptionTags::UniquePtrIntegerList>>>(
      &cpp17::as_const(cache));
  CHECK(db::get<Tags::ConstGlobalCache>(box) == &cache);
  CHECK(std::array<int, 3>{{-1, 3, 7}} ==
        db::get<OptionTags::IntegerList>(box));
  CHECK(std::array<int, 3>{{1, 5, -8}} ==
        db::get<OptionTags::UniquePtrIntegerList>(box));
  CHECK(&Parallel::get<OptionTags::IntegerList>(cache) ==
        &db::get<OptionTags::IntegerList>(box));
  CHECK(&Parallel::get<OptionTags::UniquePtrIntegerList>(cache) ==
        &db::get<OptionTags::UniquePtrIntegerList>(box));

  tuples::TaggedTuple<OptionTags::IntegerList, OptionTags::UniquePtrIntegerList>
      tuple2{};
  tuples::get<OptionTags::IntegerList>(tuple2) =
      std::array<int, 3>{{10, -3, 700}};
  tuples::get<OptionTags::UniquePtrIntegerList>(tuple2) =
      std::make_unique<std::array<int, 3>>(std::array<int, 3>{{100, -7, -300}});
  ConstGlobalCache<Metavars> cache2{std::move(tuple2)};
  db::mutate<Tags::ConstGlobalCache>(
      make_not_null(&box),
      [&cache2](
          const gsl::not_null<const Parallel::ConstGlobalCache<Metavars>**> t) {
        *t = std::addressof(cache2);
      });

  CHECK(db::get<Tags::ConstGlobalCache>(box) == &cache2);
  CHECK(std::array<int, 3>{{10, -3, 700}} ==
        db::get<OptionTags::IntegerList>(box));
  CHECK(std::array<int, 3>{{100, -7, -300}} ==
        db::get<OptionTags::UniquePtrIntegerList>(box));
  CHECK(&Parallel::get<OptionTags::IntegerList>(cache2) ==
        &db::get<OptionTags::IntegerList>(box));
  CHECK(&Parallel::get<OptionTags::UniquePtrIntegerList>(cache2) ==
        &db::get<OptionTags::UniquePtrIntegerList>(box));

  CHECK(Tags::FromConstGlobalCache<OptionTags::IntegerList>::name() ==
        "FromConstGlobalCache(IntegerList)");
  CHECK(Tags::FromConstGlobalCache<OptionTags::UniquePtrIntegerList>::name() ==
        "FromConstGlobalCache(UniquePtrIntegerList)");
}
}  // namespace Parallel
