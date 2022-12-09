// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Parallel/Algorithms/AlgorithmSingletonDeclarations.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/ResourceInfo.hpp"
#include "Parallel/Tags/ResourceInfo.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Parallel {
namespace {
template <typename Metavariables, size_t Index>
struct FakeSingleton {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  static std::string name() { return "FakeSingleton" + get_output(Index); }
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
  using simple_tags_from_options = tmpl::list<>;
};

template <size_t... Indices>
struct Metavariables {
  using component_list = tmpl::list<FakeSingleton<Metavariables, Indices>...>;
};

struct EmptyMetavars {};

template <size_t Index>
using fake_singleton = FakeSingleton<EmptyMetavars, Index>;

void test_singleton_info() {
  SingletonInfoHolder<fake_singleton<0>> info_holder{};
  CHECK(info_holder.proc() == std::nullopt);
  CHECK_FALSE(info_holder.is_exclusive());

  info_holder = TestHelpers::test_creation<
      Parallel::SingletonInfoHolder<fake_singleton<0>>>(
      "Proc: 0\n"
      "Exclusive: false\n");
  CHECK(info_holder.proc().value() == 0);
  CHECK_FALSE(info_holder.is_exclusive());

  auto serialized_info_holder = serialize_and_deserialize(info_holder);
  CHECK(serialized_info_holder == info_holder);

  info_holder = TestHelpers::test_creation<
      Parallel::SingletonInfoHolder<fake_singleton<0>>>(
      "Proc: Auto\n"
      "Exclusive: false\n");
  CHECK(info_holder.proc() == std::nullopt);
  CHECK_FALSE(info_holder.is_exclusive());

  const auto info_holder2 = TestHelpers::test_creation<
      Parallel::SingletonInfoHolder<fake_singleton<0>>>(
      "Proc: 4\n"
      "Exclusive: true\n");
  CHECK(info_holder2.proc().value() == 4);
  CHECK(info_holder2.is_exclusive());

  CHECK_FALSE(info_holder == info_holder2);
  CHECK(info_holder != info_holder2);

  CHECK_THROWS_WITH(([]() {
                      auto info_holder_error = TestHelpers::test_creation<
                          Parallel::SingletonInfoHolder<fake_singleton<0>>>(
                          "Proc: -2\n"
                          "Exclusive: true\n");
                      (void)info_holder_error;
                    })(),
                    Catch::Contains("Proc must be a non-negative integer."));
}

void test_singleton_pack() {
  const auto info0 = TestHelpers::test_creation<
      Parallel::SingletonInfoHolder<fake_singleton<0>>>(
      "Proc: Auto\n"
      "Exclusive: false\n");
  const auto info1 = TestHelpers::test_creation<
      Parallel::SingletonInfoHolder<fake_singleton<1>>>(
      "Proc: 1\n"
      "Exclusive: true\n");
  const auto info2 = TestHelpers::test_creation<
      Parallel::SingletonInfoHolder<fake_singleton<2>>>(
      "Proc: Auto\n"
      "Exclusive: true\n");

  using pack_list =
      tmpl::list<fake_singleton<0>, fake_singleton<1>, fake_singleton<2>>;
  // Make one singleton constructed with auto
  const auto singleton_pack =
      TestHelpers::test_creation<SingletonPack<pack_list>>(
          "FakeSingleton0: Auto\n"
          "FakeSingleton1:\n"
          "  Proc: 1\n"
          "  Exclusive: true\n"
          "FakeSingleton2:\n"
          "  Proc: Auto\n"
          "  Exclusive: true\n");

  const auto& pack_info0 = singleton_pack.get<fake_singleton<0>>();
  const auto& pack_info1 = singleton_pack.get<fake_singleton<1>>();
  const auto& pack_info2 = singleton_pack.get<fake_singleton<2>>();

  CHECK(info0.proc() == pack_info0.proc());
  CHECK(info0.is_exclusive() == pack_info0.is_exclusive());
  CHECK(info1.proc() == pack_info1.proc());
  CHECK(info1.is_exclusive() == pack_info1.is_exclusive());
  CHECK(info2.proc() == pack_info2.proc());
  CHECK(info2.is_exclusive() == pack_info2.is_exclusive());

  const auto serialized_pack = serialize_and_deserialize(singleton_pack);

  CHECK(serialized_pack == singleton_pack);
  CHECK_FALSE(serialized_pack != singleton_pack);
}

void test_tags() {
  using metavars = Metavariables<0>;

  TestHelpers::db::test_simple_tag<Tags::ResourceInfo<metavars>>(
      "ResourceInfo");

  Parallel::MutableGlobalCache<metavars> mutable_cache{};
  Parallel::GlobalCache<metavars> cache{{}, &mutable_cache};
  const SingletonInfoHolder<FakeSingleton<metavars, 0>> info_holder{{0}, false};

  ResourceInfo<metavars> resource_info{false, std::optional{info_holder}};
  resource_info.build_singleton_map(cache);
}

void test_single_core() {
  {
    INFO("AvoidGlobalProc0 and Singletons");
    using metavars = Metavariables<0>;
    Parallel::MutableGlobalCache<metavars> mutable_cache{};
    Parallel::GlobalCache<metavars> cache{{}, &mutable_cache};
    // Both of these should be identical since we are running on one proc
    auto resource_info_0 =
        TestHelpers::test_option_tag<OptionTags::ResourceInfo<metavars>>(
            "AvoidGlobalProc0: false\n"
            "Singletons:\n"
            "  FakeSingleton0:\n"
            "    Proc: 0\n"
            "    Exclusive: false\n");
    auto resource_info_auto =
        TestHelpers::test_option_tag<OptionTags::ResourceInfo<metavars>>(
            "AvoidGlobalProc0: false\n"
            "Singletons:\n"
            "  FakeSingleton0: Auto\n");

    resource_info_0.build_singleton_map(cache);
    resource_info_auto.build_singleton_map(cache);

    CHECK_FALSE(resource_info_0 == resource_info_auto);
    CHECK(resource_info_0 != resource_info_auto);

    CHECK_FALSE(resource_info_0.avoid_global_proc_0());
    CHECK_FALSE(resource_info_auto.avoid_global_proc_0());

    const size_t proc_0 =
        resource_info_0.template proc_for<FakeSingleton<metavars, 0>>();
    const size_t proc_auto =
        resource_info_auto.template proc_for<FakeSingleton<metavars, 0>>();

    // Only running on once proc
    CHECK(proc_0 == 0);
    CHECK(proc_auto == 0);

    const auto& info_0 =
        resource_info_0.get_singleton_info<FakeSingleton<metavars, 0>>();
    const auto& info_auto =
        resource_info_auto.get_singleton_info<FakeSingleton<metavars, 0>>();

    CHECK(info_0.proc().value() == 0);
    CHECK(info_auto.proc().value() == 0);
    CHECK_FALSE(info_0.is_exclusive());
    CHECK_FALSE(info_auto.is_exclusive());

    const auto serialized_resource_info_0 =
        serialize_and_deserialize(resource_info_0);
    const size_t serialized_proc_0 =
        resource_info_0.template proc_for<FakeSingleton<metavars, 0>>();
    const auto& serialized_info_0 =
        resource_info_0.get_singleton_info<FakeSingleton<metavars, 0>>();

    CHECK_FALSE(serialized_resource_info_0.avoid_global_proc_0());
    CHECK(proc_0 == serialized_proc_0);
    CHECK(info_0.proc() == serialized_info_0.proc());
    CHECK(info_0.is_exclusive() == serialized_info_0.is_exclusive());
  }
  {
    INFO("Singletons::Auto");
    using metavars = Metavariables<0, 1>;
    auto resource_info =
        TestHelpers::test_option_tag<OptionTags::ResourceInfo<metavars>>(
            "AvoidGlobalProc0: false\n"
            "Singletons: Auto\n");
    Parallel::MutableGlobalCache<metavars> mutable_cache{};
    Parallel::GlobalCache<metavars> cache{{}, &mutable_cache};
    resource_info.build_singleton_map(cache);
    const size_t proc_0 =
        resource_info.template proc_for<FakeSingleton<metavars, 0>>();
    const size_t proc_1 =
        resource_info.template proc_for<FakeSingleton<metavars, 1>>();
    CHECK(proc_0 == 0);
    CHECK(proc_1 == 0);

    CHECK_FALSE(resource_info.avoid_global_proc_0());
    const auto& info_0 =
        resource_info.get_singleton_info<FakeSingleton<metavars, 0>>();
    const auto& info_1 =
        resource_info.get_singleton_info<FakeSingleton<metavars, 1>>();

    CHECK(info_0.proc().value() == 0);
    CHECK(info_1.proc().value() == 0);
    CHECK_FALSE(info_0.is_exclusive());
    CHECK_FALSE(info_1.is_exclusive());
  }
}

void test_errors() {
  using metavars = Metavariables<0>;
  Parallel::MutableGlobalCache<metavars> mutable_cache{};
  Parallel::GlobalCache<metavars> cache{{}, &mutable_cache};

  // Check as many errors as we can on one proc
  CHECK_THROWS_WITH(
      ([]() {
        const auto resource_info =
            TestHelpers::test_option_tag<OptionTags::ResourceInfo<metavars>>(
                "AvoidGlobalProc0: false\n"
                "Singletons:\n"
                "  FakeSingleton0:\n"
                "    Proc: -2\n"
                "    Exclusive: false\n");
      })(),
      Catch::Contains("Proc must be a non-negative integer."));

  CHECK_THROWS_WITH(
      ([]() {
        const auto resource_info =
            TestHelpers::test_option_tag<OptionTags::ResourceInfo<metavars>>(
                "AvoidGlobalProc0: true\n"
                "Singletons:\n"
                "  FakeSingleton0:\n"
                "    Proc: 0\n"
                "    Exclusive: true\n");
      })(),
      Catch::Contains("A singleton has requested to be exclusively on proc 0, "
                      "but the AvoidGlobalProc0 option is also set to true."));

  CHECK_THROWS_WITH(
      ([]() {
        const auto resource_info = TestHelpers::test_option_tag<
            OptionTags::ResourceInfo<Metavariables<0, 1>>>(
            "AvoidGlobalProc0: false\n"
            "Singletons:\n"
            "  FakeSingleton0:\n"
            "    Proc: 0\n"
            "    Exclusive: true\n"
            "  FakeSingleton1:\n"
            "    Proc: 0\n"
            "    Exclusive: true\n");
      })(),
      Catch::Contains(
          "Two singletons have requested to be on proc 0, but at least one of "
          "them has requested to be exclusively on this proc."));

  CHECK_THROWS_WITH(
      ([&cache]() {
        auto resource_info =
            TestHelpers::test_option_tag<OptionTags::ResourceInfo<metavars>>(
                "AvoidGlobalProc0: false\n"
                "Singletons:\n"
                "  FakeSingleton0:\n"
                "    Proc: 2\n"
                "    Exclusive: false\n");
        resource_info.build_singleton_map(cache);
      })(),
      Catch::Contains("is beyond the last proc"));

  CHECK_THROWS_WITH(
      ([&cache]() {
        auto resource_info =
            TestHelpers::test_option_tag<OptionTags::ResourceInfo<metavars>>(
                "AvoidGlobalProc0: false\n"
                "Singletons:\n"
                "  FakeSingleton0:\n"
                "    Proc: 0\n"
                "    Exclusive: true\n");
        resource_info.build_singleton_map(cache);
      })(),
      Catch::Contains(
          "The total number of cores requested is less than or equal to the "
          "number of cores that requested to be exclusive, i.e. without"));

  CHECK_THROWS_WITH(
      ([]() {
        auto resource_info =
            TestHelpers::test_option_tag<OptionTags::ResourceInfo<metavars>>(
                "AvoidGlobalProc0: true\n"
                "Singletons:\n"
                "  FakeSingleton0:\n"
                "    Proc: 0\n"
                "    Exclusive: false\n");
        [[maybe_unused]] const size_t proc =
            resource_info.template proc_for<FakeSingleton<metavars, 0>>();
      })(),
      Catch::Contains("The singleton map has not been built yet. You must call "
                      "build_singleton_map() before you call this function."));

  CHECK_THROWS_WITH(
      ([]() {
        auto resource_info =
            TestHelpers::test_option_tag<OptionTags::ResourceInfo<metavars>>(
                "AvoidGlobalProc0: true\n"
                "Singletons:\n"
                "  FakeSingleton0:\n"
                "    Proc: 0\n"
                "    Exclusive: false\n");
        [[maybe_unused]] const auto& procs_to_ignore =
            resource_info.procs_to_ignore();
      })(),
      Catch::Contains("The singleton map has not been built yet. You must call "
                      "build_singleton_map() before you call this function."));

  CHECK_THROWS_WITH(
      ([]() {
        auto resource_info =
            TestHelpers::test_option_tag<OptionTags::ResourceInfo<metavars>>(
                "AvoidGlobalProc0: true\n"
                "Singletons:\n"
                "  FakeSingleton0:\n"
                "    Proc: 0\n"
                "    Exclusive: false\n");
        [[maybe_unused]] const auto& procs_to_ignore =
            resource_info.procs_available_for_elements();
      })(),
      Catch::Contains("The singleton map has not been built yet. You must call "
                      "build_singleton_map() before you call this function."));

  CHECK_THROWS_WITH(
      ([]() {
        auto resource_info =
            TestHelpers::test_option_tag<OptionTags::ResourceInfo<metavars>>(
                "AvoidGlobalProc0: true\n"
                "Singletons:\n"
                "  FakeSingleton0:\n"
                "    Proc: 0\n"
                "    Exclusive: false\n");
        [[maybe_unused]] const auto singleton_info =
            resource_info.get_singleton_info<FakeSingleton<metavars, 0>>();
      })(),
      Catch::Contains("The singleton map has not been built yet. You must call "
                      "build_singleton_map() before you call this function."));
}

template <typename Metavariables>
Parallel::ResourceInfo<Metavariables> create_resource_info(
    const bool avoid_global_proc_0,
    const std::vector<std::pair<bool, int>>& singletons) {
  std::string option_str =
      "AvoidGlobalProc0: " + (avoid_global_proc_0 ? "true"s : "false"s) + "\n";
  option_str += "Singletons:\n";
  for (size_t i = 0; i < singletons.size(); i++) {
    const bool exclusive = singletons[i].first;
    const int proc = singletons[i].second;
    option_str += "  FakeSingleton" + get_output(i) + ":\n";
    option_str +=
        "    Proc: " + (proc == -1 ? "Auto"s : get_output(proc)) + "\n";
    option_str += "    Exclusive: " + (exclusive ? "true"s : "false"s) + "\n";
  }

  return serialize_and_deserialize(
      TestHelpers::test_option_tag<OptionTags::ResourceInfo<Metavariables>>(
          option_str));
}

constexpr size_t num_singletons = 7;
using metavars = Metavariables<0, 1, 2, 3, 4, 5, 6>;
template <size_t Index>
using component = FakeSingleton<metavars, Index>;

void check_resource_info(
    const Parallel::GlobalCache<metavars>& cache, bool avoid_global_proc_0,
    const std::vector<std::pair<bool, int>>& singletons,
    const std::vector<std::pair<bool, int>>& expected_singletons) {
  auto resource_info =
      create_resource_info<metavars>(avoid_global_proc_0, singletons);
  resource_info.build_singleton_map(cache);
  const size_t num_procs = Parallel::number_of_procs<size_t>(cache);

  std::unordered_set<size_t> expected_exclusive_procs{};
  std::set<size_t> expected_procs_available_for_elements{};
  for (auto& exclusive_and_proc : expected_singletons) {
    if (exclusive_and_proc.first) {
      expected_exclusive_procs.insert(
          static_cast<size_t>(exclusive_and_proc.second));
    }
  }

  if (avoid_global_proc_0) {
    expected_exclusive_procs.insert(0);
  }

  // Construct the expected procs with elements slightly differently than inside
  // ResourceInfo just to check we did it correctly. They should be equivalent
  for (size_t i = 0; i < num_procs; i++) {
    expected_procs_available_for_elements.insert(i);
  }
  for (size_t proc : expected_exclusive_procs) {
    expected_procs_available_for_elements.erase(proc);
  }

  const auto& procs_to_ignore = resource_info.procs_to_ignore();
  const auto& procs_available_for_elements =
      resource_info.procs_available_for_elements();

  CHECK(resource_info.avoid_global_proc_0() == avoid_global_proc_0);
  CHECK(procs_to_ignore.size() == expected_exclusive_procs.size());
  for (auto& exclusive_proc : expected_exclusive_procs) {
    CHECK(procs_to_ignore.count(exclusive_proc) == 1);
  }
  CHECK(procs_available_for_elements.size() ==
        num_procs - expected_exclusive_procs.size());
  CHECK(procs_available_for_elements == expected_procs_available_for_elements);

  tmpl::for_each<tmpl::range<size_t, 0, num_singletons>>(
      [&expected_singletons, &resource_info](const auto size_holder) {
        constexpr size_t index =
            std::decay_t<decltype(size_holder)>::type::value;
        INFO("Index: " + get_output(index));
        auto singleton_info =
            resource_info.get_singleton_info<component<index>>();
        CHECK(singleton_info.is_exclusive() ==
              expected_singletons[index].first);
        CHECK(static_cast<int>(singleton_info.proc().value()) ==
              expected_singletons[index].second);
        CHECK(resource_info.template proc_for<component<index>>() ==
              static_cast<size_t>(expected_singletons[index].second));
      });
}

template <typename Gen>
void test_single_node_multi_core(const gsl::not_null<Gen*> gen) {
  Parallel::MutableGlobalCache<metavars> mutable_cache{};
  // 1 node, 3 procs per node
  Parallel::GlobalCache<metavars> cache{{}, &mutable_cache, {3}};

  INFO("1 node, 3 procs per node");

  {
    INFO("AvoidGlobalProc0 false; All singletons Auto and not exclusive");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] = std::pair<bool, int>{false, -1};
    }
    const std::vector<std::pair<bool, int>> expected{
        {false, 0}, {false, 0}, {false, 0}, {false, 1},
        {false, 1}, {false, 2}, {false, 2}};

    check_resource_info(cache, false, singletons, expected);
  }
  {
    INFO("AvoidGlobalProc0 true; All singletons Auto and not exclusive");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] = std::pair<bool, int>{false, -1};
    }
    const std::vector<std::pair<bool, int>> expected{
        {false, 1}, {false, 1}, {false, 1}, {false, 1},
        {false, 2}, {false, 2}, {false, 2}};

    check_resource_info(cache, true, singletons, expected);
  }
  {
    INFO("AvoidGlobalProc0 false; One singleton exclusive Auto");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] = std::pair<bool, int>{i == 4, -1};
    }
    const std::vector<std::pair<bool, int>> expected{
        {false, 1}, {false, 1}, {false, 1}, {false, 2},
        {true, 0},  {false, 2}, {false, 2}};

    check_resource_info(cache, false, singletons, expected);
  }
  {
    INFO("AvoidGlobalProc0 true; One singleton exclusive Auto");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] = std::pair<bool, int>{i == 6, -1};
    }
    const std::vector<std::pair<bool, int>> expected{
        {false, 2}, {false, 2}, {false, 2}, {false, 2},
        {false, 2}, {false, 2}, {true, 1}};

    check_resource_info(cache, true, singletons, expected);
  }
  {
    INFO("AvoidGlobalProc0 false; All singletons requested and non-exclusive");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    std::uniform_int_distribution<int> dist{0, 2};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] = std::pair<bool, int>{false, dist(*gen)};
    }

    check_resource_info(cache, false, singletons, singletons);
  }
  {
    INFO("AvoidGlobalProc0 false; Two singletons exclusive, both Auto");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] = std::pair<bool, int>{i == 2 or i == 4, -1};
    }
    const std::vector<std::pair<bool, int>> expected{
        {false, 2}, {false, 2}, {true, 0}, {false, 2},
        {true, 1},  {false, 2}, {false, 2}};

    check_resource_info(cache, false, singletons, expected);
  }
  {
    INFO(
        "AvoidGlobalProc0 false; Two singletons exclusive, one Auto one "
        "requested");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] = std::pair<bool, int>{i == 2 or i == 4, i == 2 ? 1 : -1};
    }
    const std::vector<std::pair<bool, int>> expected{
        {false, 2}, {false, 2}, {true, 1}, {false, 2},
        {true, 0},  {false, 2}, {false, 2}};

    check_resource_info(cache, false, singletons, expected);
  }
  {
    INFO("AvoidGlobalProc0 false; Two singletons exclusive, both requested");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] =
          std::pair<bool, int>{i == 2 or i == 4, i == 2 ? 0 : i == 4 ? 2 : -1};
    }
    const std::vector<std::pair<bool, int>> expected{
        {false, 1}, {false, 1}, {true, 0}, {false, 1},
        {true, 2},  {false, 1}, {false, 1}};

    check_resource_info(cache, false, singletons, expected);
  }
  {
    INFO(
        "AvoidGlobalProc0 false; One singleton exclusive Auto, one "
        "nonexclusive requested");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] = std::pair<bool, int>{i == 4, i == 5 ? 0 : -1};
    }
    const std::vector<std::pair<bool, int>> expected{
        {false, 0}, {false, 0}, {false, 2}, {false, 2},
        {true, 1},  {false, 0}, {false, 2}};

    check_resource_info(cache, false, singletons, expected);
  }
  {
    INFO("AvoidGlobalProc0 false; All random");
    const std::vector<std::pair<bool, int>> singletons{
        {false, 2}, {false, 2},  {false, -1}, {false, 1},
        {true, -1}, {false, -1}, {false, -1}};
    const std::vector<std::pair<bool, int>> expected{
        {false, 2}, {false, 2}, {false, 1}, {false, 1},
        {true, 0},  {false, 1}, {false, 2}};

    check_resource_info(cache, false, singletons, expected);
  }
}

template <typename Gen>
void test_multi_node_multi_core(const gsl::not_null<Gen*> gen) {
  Parallel::MutableGlobalCache<metavars> mutable_cache{};
  // 3 nodes, 2 procs per node
  Parallel::GlobalCache<metavars> cache{{}, &mutable_cache, {2, 2, 2}};

  INFO("3 nodes, 2 procs per node");

  {
    INFO("AvoidGlobalProc0 false; All singletons Auto and not exclusive");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] = std::pair<bool, int>{false, -1};
    }
    const std::vector<std::pair<bool, int>> expected{
        {false, 0}, {false, 0}, {false, 1}, {false, 2},
        {false, 3}, {false, 4}, {false, 5}};

    check_resource_info(cache, false, singletons, expected);
  }
  {
    INFO("AvoidGlobalProc0 true; All singletons Auto and not exclusive");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] = std::pair<bool, int>{false, -1};
    }
    const std::vector<std::pair<bool, int>> expected{
        {false, 1}, {false, 1}, {false, 1}, {false, 2},
        {false, 3}, {false, 4}, {false, 5}};

    check_resource_info(cache, true, singletons, expected);
  }
  {
    INFO(
        "AvoidGlobalProc0 false; All singletons specific procs and not "
        "exclusive");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    std::uniform_int_distribution<int> dist{0, 5};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] = std::pair<bool, int>{false, dist(*gen)};
    }

    check_resource_info(cache, false, singletons, singletons);
  }
  {
    INFO(
        "AvoidGlobalProc0 true; All singletons specific procs and not "
        "exclusive");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    std::uniform_int_distribution<int> dist{1, 5};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] = std::pair<bool, int>{false, dist(*gen)};
    }

    check_resource_info(cache, true, singletons, singletons);
  }
  {
    INFO("AvoidGlobalProc0 false; Mix and match singletons #1: Exclusive Auto");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] = std::pair<bool, int>{i < 3, -1};
    }
    const std::vector<std::pair<bool, int>> expected{
        {true, 0},  {true, 2},  {true, 4}, {false, 1},
        {false, 1}, {false, 3}, {false, 5}};

    check_resource_info(cache, false, singletons, expected);
  }
  {
    INFO(
        "AvoidGlobalProc0 false; Mix and match singletons #2: Exclusive "
        "specific");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    std::uniform_int_distribution<int> dist{3, 5};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] =
          std::pair<bool, int>{i < 3, i < 3 ? static_cast<int>(i) : dist(*gen)};
    }

    check_resource_info(cache, false, singletons, singletons);
  }
  {
    INFO("AvoidGlobalProc0 true; Mix and match singletons #3: Exclusive Auto");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] = std::pair<bool, int>{i < 3, -1};
    }
    const std::vector<std::pair<bool, int>> expected{
        {true, 1},  {true, 2},  {true, 4}, {false, 3},
        {false, 3}, {false, 5}, {false, 5}};

    check_resource_info(cache, true, singletons, expected);
  }
  {
    INFO(
        "AvoidGlobalProc0 true; Mix and match singletons #4: Exclusive "
        "specific");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    std::uniform_int_distribution<int> dist{4, 5};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] = std::pair<bool, int>{
          i < 3, i < 3 ? static_cast<int>(i) + 1 : dist(*gen)};
    }

    check_resource_info(cache, true, singletons, singletons);
  }
  {
    INFO("AvoidGlobalProc0 false; Mix and match singletons #5: All random");
    std::vector<std::pair<bool, int>> singletons{
        {false, -1}, {false, -1}, {false, 3}, {true, 2},
        {false, 3},  {true, -1},  {false, -1}};
    std::vector<std::pair<bool, int>> expected{
        {false, 1}, {false, 4}, {false, 3}, {true, 2},
        {false, 3}, {true, 0},  {false, 5}};

    check_resource_info(cache, false, singletons, expected);
  }
  {
    INFO("AvoidGlobalProc0 true; Mix and match singletons #6: All random");
    std::vector<std::pair<bool, int>> singletons{
        {true, 1},  {false, 2},  {false, -1}, {false, 5},
        {true, -1}, {false, -1}, {false, -1}};
    std::vector<std::pair<bool, int>> expected{
        {true, 1}, {false, 2}, {false, 2}, {false, 5},
        {true, 3}, {false, 4}, {false, 4}};

    check_resource_info(cache, true, singletons, expected);
  }
  {
    INFO("AvoidGlobalProc0 false; Mix and match singletons #7: All random");
    std::vector<std::pair<bool, int>> singletons{
        {true, 1},  {false, -1}, {true, 3},  {false, -1},
        {true, -1}, {true, 5},   {false, -1}};
    std::vector<std::pair<bool, int>> expected{
        {true, 1}, {false, 2}, {true, 3}, {false, 2},
        {true, 0}, {true, 5},  {false, 4}};

    check_resource_info(cache, false, singletons, expected);
  }
  {
    INFO("AvoidGlobalProc0 false; Mix and match singletons #8: All random");
    std::vector<std::pair<bool, int>> singletons{
        {true, 1},  {false, 0}, {true, 3},  {false, -1},
        {true, -1}, {true, 5},  {false, -1}};
    std::vector<std::pair<bool, int>> expected{
        {true, 1}, {false, 0}, {true, 3}, {false, 4},
        {true, 2}, {true, 5},  {false, 4}};

    check_resource_info(cache, false, singletons, expected);
  }
  {
    INFO(
        "AvoidGlobalProc0 false; Mix and match singletons #9: Different number "
        "of procs on node");
    Parallel::MutableGlobalCache<metavars> local_mutable_cache{};
    // 3 nodes, (1,2,2) procs on each node
    Parallel::GlobalCache<metavars> local_cache{
        {}, &local_mutable_cache, {1, 2, 2}};
    std::vector<std::pair<bool, int>> singletons{
        {false, -1}, {true, -1}, {false, -1}, {false, -1},
        {true, -1},  {true, 0},  {true, 1}};
    std::vector<std::pair<bool, int>> expected{
        {false, 4}, {true, 2}, {false, 4}, {false, 4},
        {true, 3},  {true, 0}, {true, 1}};

    check_resource_info(local_cache, false, singletons, expected);
  }
}

template <typename Gen>
void test_multi_node_multi_core_large(const gsl::not_null<Gen*> gen) {
  Parallel::MutableGlobalCache<metavars> mutable_cache{};
  // 8 nodes, 25 procs per node
  const size_t num_nodes = 8;
  const size_t num_procs_per_node = 25;
  const size_t num_procs = num_nodes * num_procs_per_node;
  Parallel::GlobalCache<metavars> cache{
      {}, &mutable_cache, std::vector<size_t>(num_nodes, num_procs_per_node)};

  INFO("8 nodes, 25 procs per node");

  {
    INFO("AvoidGlobalProc0 false; All singletons Auto and not exclusive");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] = std::pair<bool, int>{false, -1};
    }
    const std::vector<std::pair<bool, int>> expected{
        {false, 0},   {false, 25},  {false, 50}, {false, 75},
        {false, 100}, {false, 125}, {false, 150}};

    check_resource_info(cache, false, singletons, expected);
  }
  {
    INFO("AvoidGlobalProc0 true; All singletons Auto and not exclusive");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] = std::pair<bool, int>{false, -1};
    }
    const std::vector<std::pair<bool, int>> expected{
        {false, 1},   {false, 25},  {false, 50}, {false, 75},
        {false, 100}, {false, 125}, {false, 150}};

    check_resource_info(cache, true, singletons, expected);
  }
  {
    INFO(
        "AvoidGlobalProc0 false; All singletons specific procs and not "
        "exclusive");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    std::uniform_int_distribution<int> dist{0, num_procs - 1};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] = std::pair<bool, int>{false, dist(*gen)};
    }

    check_resource_info(cache, false, singletons, singletons);
  }
  {
    INFO(
        "AvoidGlobalProc0 true; All singletons specific procs and not "
        "exclusive");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    std::uniform_int_distribution<int> dist{1, num_procs - 1};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] = std::pair<bool, int>{false, dist(*gen)};
    }

    check_resource_info(cache, true, singletons, singletons);
  }
  {
    INFO("AvoidGlobalProc0 false; Mix and match singletons #1: Exclusive Auto");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] = std::pair<bool, int>{i < 3, -1};
    }
    const std::vector<std::pair<bool, int>> expected{
        {true, 0},    {true, 25},   {true, 50},  {false, 75},
        {false, 100}, {false, 125}, {false, 150}};

    check_resource_info(cache, false, singletons, expected);
  }
  {
    INFO(
        "AvoidGlobalProc0 false; Mix and match singletons #2: Exclusive "
        "specific");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    std::unordered_set<int> used_procs{};
    std::uniform_int_distribution<int> dist_exclusive{0, 100};
    std::uniform_int_distribution<int> dist_nonexclusive{101, num_procs - 1};
    for (size_t i = 0; i < num_singletons; i++) {
      int exclusive_proc = dist_exclusive(*gen);
      while (used_procs.count(exclusive_proc) != 0) {
        exclusive_proc = dist_exclusive(*gen);
      }
      used_procs.insert(exclusive_proc);
      singletons[i] = std::pair<bool, int>{
          i < 3, i < 3 ? exclusive_proc : dist_nonexclusive(*gen)};
    }

    check_resource_info(cache, false, singletons, singletons);
  }
  {
    INFO("AvoidGlobalProc0 true; Mix and match singletons #3: Exclusive Auto");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    for (size_t i = 0; i < num_singletons; i++) {
      singletons[i] = std::pair<bool, int>{i < 3, -1};
    }
    const std::vector<std::pair<bool, int>> expected{
        {true, 1},    {true, 25},   {true, 50},  {false, 75},
        {false, 100}, {false, 125}, {false, 150}};

    check_resource_info(cache, true, singletons, expected);
  }
  {
    INFO(
        "AvoidGlobalProc0 true; Mix and match singletons #4: Exclusive "
        "specific");
    std::vector<std::pair<bool, int>> singletons{num_singletons};
    std::unordered_set<int> used_procs{};
    std::uniform_int_distribution<int> dist_exclusive{1, 100};
    std::uniform_int_distribution<int> dist_nonexclusive{101, num_procs - 1};
    for (size_t i = 0; i < num_singletons; i++) {
      int exclusive_proc = dist_exclusive(*gen);
      while (used_procs.count(exclusive_proc) != 0) {
        exclusive_proc = dist_exclusive(*gen);
      }
      used_procs.insert(exclusive_proc);
      singletons[i] = std::pair<bool, int>{
          i < 3, i < 3 ? exclusive_proc : dist_nonexclusive(*gen)};
    }

    check_resource_info(cache, true, singletons, singletons);
  }
  {
    INFO("AvoidGlobalProc0 false; Mix and match singletons #5: All random");
    std::vector<std::pair<bool, int>> singletons{
        {false, 57}, {false, 56}, {true, -1}, {true, 149},
        {false, -1}, {false, -1}, {false, -1}};
    std::vector<std::pair<bool, int>> expected{
        {false, 57}, {false, 56}, {true, 0},   {true, 149},
        {false, 25}, {false, 75}, {false, 100}};

    check_resource_info(cache, false, singletons, expected);
  }
  {
    INFO("AvoidGlobalProc0 true; Mix and match singletons #6: All random");
    std::vector<std::pair<bool, int>> singletons{
        {true, -1}, {true, 76}, {false, 77}, {false, 77},
        {true, -1}, {true, -1}, {true, -1}};
    std::vector<std::pair<bool, int>> expected{
        {true, 1},  {true, 76}, {false, 77}, {false, 77},
        {true, 25}, {true, 50}, {true, 100}};

    check_resource_info(cache, true, singletons, expected);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.ResourceInfo", "[Unit][Parallel]") {
  MAKE_GENERATOR(gen)
  test_singleton_info();
  test_singleton_pack();
  test_tags();
  test_single_core();
  test_single_node_multi_core(make_not_null(&gen));
  test_multi_node_multi_core(make_not_null(&gen));
  test_multi_node_multi_core_large(make_not_null(&gen));
  test_errors();
}
}  // namespace Parallel
