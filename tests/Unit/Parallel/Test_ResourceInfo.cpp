// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <string>

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/Algorithms/AlgorithmSingletonDeclarations.hpp"
#include "Parallel/ResourceInfo.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

namespace Parallel {
namespace {
template <size_t Index>
struct FakeSingleton {
  using chare_type = Parallel::Algorithms::Singleton;
  static std::string name() { return "FakeSingleton" + get_output(Index); }
};

void test_singleton_info() {
  SingletonInfoHolder<FakeSingleton<0>> info_holder{};
  CHECK(info_holder.proc() == std::nullopt);
  CHECK_FALSE(info_holder.is_exclusive());

  info_holder = TestHelpers::test_creation<
      Parallel::SingletonInfoHolder<FakeSingleton<0>>>(
      "Proc: 0\n"
      "Exclusive: false\n");
  CHECK(info_holder.proc().value() == 0);
  CHECK_FALSE(info_holder.is_exclusive());

  auto serialized_info_holder = serialize_and_deserialize(info_holder);
  CHECK(info_holder.proc().value() == serialized_info_holder.proc().value());
  CHECK(info_holder.is_exclusive() == serialized_info_holder.is_exclusive());

  info_holder = TestHelpers::test_creation<
      Parallel::SingletonInfoHolder<FakeSingleton<0>>>(
      "Proc: Auto\n"
      "Exclusive: false\n");
  CHECK(info_holder.proc() == std::nullopt);
  CHECK_FALSE(info_holder.is_exclusive());

  info_holder = TestHelpers::test_creation<
      Parallel::SingletonInfoHolder<FakeSingleton<0>>>(
      "Proc: 4\n"
      "Exclusive: true\n");
  CHECK(info_holder.proc().value() == 4);
  CHECK(info_holder.is_exclusive());

  CHECK_THROWS_WITH(([]() {
                      auto info_holder_error = TestHelpers::test_creation<
                          Parallel::SingletonInfoHolder<FakeSingleton<0>>>(
                          "Proc: -2\n"
                          "Exclusive: true\n");
                      (void)info_holder_error;
                    })(),
                    Catch::Contains("Proc must be a non-negative integer."));
}

void test_singleton_pack() {
  const auto info0 = TestHelpers::test_creation<
      Parallel::SingletonInfoHolder<FakeSingleton<0>>>(
      "Proc: 0\n"
      "Exclusive: false\n");
  const auto info1 = TestHelpers::test_creation<
      Parallel::SingletonInfoHolder<FakeSingleton<1>>>(
      "Proc: 1\n"
      "Exclusive: true\n");
  const auto info2 = TestHelpers::test_creation<
      Parallel::SingletonInfoHolder<FakeSingleton<2>>>(
      "Proc: Auto\n"
      "Exclusive: true\n");

  using pack_list =
      tmpl::list<FakeSingleton<0>, FakeSingleton<1>, FakeSingleton<2>>;
  const auto singleton_pack =
      TestHelpers::test_creation<SingletonPack<pack_list>>(
          "FakeSingleton0:\n"
          "  Proc: 0\n"
          "  Exclusive: false\n"
          "FakeSingleton1:\n"
          "  Proc: 1\n"
          "  Exclusive: true\n"
          "FakeSingleton2:\n"
          "  Proc: Auto\n"
          "  Exclusive: true\n");

  const auto& pack_info0 = singleton_pack.get<FakeSingleton<0>>();
  const auto& pack_info1 = singleton_pack.get<FakeSingleton<1>>();
  const auto& pack_info2 = singleton_pack.get<FakeSingleton<2>>();

  CHECK(info0.proc() == pack_info0.proc());
  CHECK(info0.is_exclusive() == pack_info0.is_exclusive());
  CHECK(info1.proc() == pack_info1.proc());
  CHECK(info1.is_exclusive() == pack_info1.is_exclusive());
  CHECK(info2.proc() == pack_info2.proc());
  CHECK(info2.is_exclusive() == pack_info2.is_exclusive());

  const auto serialized_pack = serialize_and_deserialize(singleton_pack);
  const auto& serialized_pack_info0 = serialized_pack.get<FakeSingleton<0>>();
  const auto& serialized_pack_info1 = serialized_pack.get<FakeSingleton<1>>();
  const auto& serialized_pack_info2 = serialized_pack.get<FakeSingleton<2>>();

  CHECK(serialized_pack_info0.proc() == pack_info0.proc());
  CHECK(serialized_pack_info0.is_exclusive() == pack_info0.is_exclusive());
  CHECK(serialized_pack_info1.proc() == pack_info1.proc());
  CHECK(serialized_pack_info1.is_exclusive() == pack_info1.is_exclusive());
  CHECK(serialized_pack_info2.proc() == pack_info2.proc());
  CHECK(serialized_pack_info2.is_exclusive() == pack_info2.is_exclusive());

  CHECK_THROWS_WITH(
      ([]() {
        const SingletonPack<tmpl::list<>> empty_pack{};
        const auto serialized_empty_pack =
            serialize_and_deserialize(empty_pack);
        const auto& throws_error =
            serialized_empty_pack.get<FakeSingleton<0>>();
        (void)throws_error;
      })(),
      Catch::Contains("Cannot call the get() member of a SingletonPack with an "
                      "empty component list."));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.ResourceInfo", "[Unit][Parallel]") {
  test_singleton_info();
  test_singleton_pack();
}
}  // namespace Parallel
