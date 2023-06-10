// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Parallel {
namespace Tags {
struct IntegerList : db::SimpleTag {
  using type = std::array<int, 3>;
};

struct UniquePtrIntegerListBase : db::BaseTag {};

struct UniquePtrIntegerList : UniquePtrIntegerListBase, db::SimpleTag {
  using type = std::unique_ptr<std::array<int, 3>>;
};
}  // namespace Tags

namespace {
struct Metavars {
  using const_global_cache_tags =
      tmpl::list<Tags::IntegerList, Tags::UniquePtrIntegerList>;
  using component_list = tmpl::list<>;
};

struct EmptyMetavars {
  using component_list = tmpl::list<>;
};

void test_mutable_cache_proxy_error() {
  CHECK_THROWS_WITH(
      ([]() {
        MutableGlobalCache<EmptyMetavars> mutable_cache{
            tuples::TaggedTuple<>{}};
        GlobalCache<EmptyMetavars> cache{tuples::TaggedTuple<>{},
                                         &mutable_cache};

        auto mutable_cache_proxy = cache.mutable_global_cache_proxy();
      })(),
      Catch::Contains(
          "Cannot get a proxy to the mutable global cache because the proxy "
          "isn't set."));
}

void test_mem_monitor_entry_method_error() {
  CHECK_THROWS_WITH(
      ([]() {
        MutableGlobalCache<EmptyMetavars> mutable_cache{
            tuples::TaggedTuple<>{}};
        CProxy_GlobalCache<EmptyMetavars> empty_cache_proxy{};

        mutable_cache.compute_size_for_memory_monitor(empty_cache_proxy, 0.0);
      })(),
      Catch::Contains(
          "MutableGlobalCache::compute_size_for_memory_monitor can only be "
          "called if the MemoryMonitor is in the component list in the "
          "metavariables.\n"));

  CHECK_THROWS_WITH(
      ([]() {
        MutableGlobalCache<EmptyMetavars> mutable_cache{
            tuples::TaggedTuple<>{}};
        GlobalCache<EmptyMetavars> empty_cache{tuples::TaggedTuple<>{},
                                               &mutable_cache};

        empty_cache.compute_size_for_memory_monitor(0.0);
      })(),
      Catch::Contains(
          "GlobalCache::compute_size_for_memory_monitor can only be called if "
          "the MemoryMonitor is in the component list in the "
          "metavariables.\n"));
}

void test_resource_info_error() {
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      ([]() {
        MutableGlobalCache<EmptyMetavars> mutable_cache{
            tuples::TaggedTuple<>{}};
        GlobalCache<EmptyMetavars> empty_cache{tuples::TaggedTuple<>{},
                                               &mutable_cache};

        empty_cache.set_resource_info(ResourceInfo<EmptyMetavars>{});
        empty_cache.set_resource_info(ResourceInfo<EmptyMetavars>{});
      })(),
      Catch::Contains("Can only set the resource info once"));
#endif
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.GlobalCacheDataBox", "[Unit][Parallel]") {
  test_mutable_cache_proxy_error();
  test_mem_monitor_entry_method_error();
  test_resource_info_error();

  tuples::TaggedTuple<Tags::IntegerList, Tags::UniquePtrIntegerList> tuple{};
  tuples::get<Tags::IntegerList>(tuple) = std::array<int, 3>{{-1, 3, 7}};
  tuples::get<Tags::UniquePtrIntegerList>(tuple) =
      std::make_unique<std::array<int, 3>>(std::array<int, 3>{{1, 5, -8}});
  MutableGlobalCache<Metavars> mutable_cache{tuples::TaggedTuple<>{}};
  GlobalCache<Metavars> cache{std::move(tuple), &mutable_cache};
  Parallel::ResourceInfo<Metavars> resource_info{false};
  resource_info.build_singleton_map(cache);
  cache.set_resource_info(resource_info);
  auto box = db::create<
      db::AddSimpleTags<Tags::GlobalCacheImpl<Metavars>>,
      db::AddComputeTags<Tags::FromGlobalCache<Tags::IntegerList>,
                         Tags::FromGlobalCache<Tags::UniquePtrIntegerList>,
                         Tags::ResourceInfoReference<Metavars>>>(&cache);
  CHECK(db::get<Tags::GlobalCache>(box) == &cache);
  CHECK(std::array<int, 3>{{-1, 3, 7}} == db::get<Tags::IntegerList>(box));
  CHECK(std::array<int, 3>{{1, 5, -8}} ==
        db::get<Tags::UniquePtrIntegerList>(box));
  CHECK(std::array<int, 3>{{1, 5, -8}} ==
        db::get<Tags::UniquePtrIntegerListBase>(box));
  CHECK(cache.get_resource_info() == resource_info);
  CHECK(cache.get_resource_info() ==
        db::get<Tags::ResourceInfo<Metavars>>(box));
  CHECK(&Parallel::get<Tags::IntegerList>(cache) ==
        &db::get<Tags::IntegerList>(box));
  CHECK(&Parallel::get<Tags::UniquePtrIntegerList>(cache) ==
        &db::get<Tags::UniquePtrIntegerList>(box));
  CHECK(&Parallel::get<Tags::UniquePtrIntegerList>(cache) ==
        &db::get<Tags::UniquePtrIntegerListBase>(box));
  CHECK(&cache.get_resource_info() ==
        &db::get<Tags::ResourceInfo<Metavars>>(box));

  tuples::TaggedTuple<Tags::IntegerList, Tags::UniquePtrIntegerList> tuple2{};
  tuples::get<Tags::IntegerList>(tuple2) = std::array<int, 3>{{10, -3, 700}};
  tuples::get<Tags::UniquePtrIntegerList>(tuple2) =
      std::make_unique<std::array<int, 3>>(std::array<int, 3>{{100, -7, -300}});
  MutableGlobalCache<Metavars> mutable_cache2{tuples::TaggedTuple<>{}};
  GlobalCache<Metavars> cache2{std::move(tuple2), &mutable_cache2};
  db::mutate<Tags::GlobalCache>(
      [&cache2](const gsl::not_null<Parallel::GlobalCache<Metavars>**> t) {
        *t = std::addressof(cache2);
      },
      make_not_null(&box));

  CHECK(db::get<Tags::GlobalCache>(box) == &cache2);
  CHECK(std::array<int, 3>{{10, -3, 700}} == db::get<Tags::IntegerList>(box));
  CHECK(std::array<int, 3>{{100, -7, -300}} ==
        db::get<Tags::UniquePtrIntegerList>(box));
  CHECK(std::array<int, 3>{{100, -7, -300}} ==
        db::get<Tags::UniquePtrIntegerListBase>(box));
  CHECK(&Parallel::get<Tags::IntegerList>(cache2) ==
        &db::get<Tags::IntegerList>(box));
  CHECK(&Parallel::get<Tags::UniquePtrIntegerList>(cache2) ==
        &db::get<Tags::UniquePtrIntegerList>(box));
  CHECK(&Parallel::get<Tags::UniquePtrIntegerList>(cache2) ==
        &db::get<Tags::UniquePtrIntegerListBase>(box));

  TestHelpers::db::test_base_tag<Tags::GlobalCache>("GlobalCache");
  TestHelpers::db::test_simple_tag<Tags::GlobalCacheImpl<Metavars>>(
      "GlobalCache");
  TestHelpers::db::test_reference_tag<Tags::FromGlobalCache<Tags::IntegerList>>(
      "IntegerList");
  TestHelpers::db::test_reference_tag<
      Tags::FromGlobalCache<Tags::UniquePtrIntegerList>>(
      "UniquePtrIntegerList");
  TestHelpers::db::test_reference_tag<Tags::ResourceInfoReference<Metavars>>(
      "ResourceInfo");
}
}  // namespace Parallel
