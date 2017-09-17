// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include "tests/Unit/Parallel/Test_ConstGlobalCache.hpp"

#include <catch.hpp>
#include <exception>
#include <memory>

#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Parallel/Abort.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Exit.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

struct name {
  using type = std::string;
};

struct age {
  using type = int;
};

struct height {
  using type = double;
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
class Shape : public PUP::able {
 public:
  Shape() = default;
  virtual size_t number_of_sides() const = 0;
  // clang-tidy: internal charm++ warnings
  WRAPPED_PUPable_abstract(Shape);  // NOLINT
};

class Triangle : public Shape {
 public:
  Triangle() = default;
  explicit Triangle(CkMigrateMessage* /*m*/) {}
  size_t number_of_sides() const noexcept final { return 3; }
  // clang-tidy: internal charm++ warnings
  WRAPPED_PUPable_decl_base_template(Shape,  // NOLINT
                                     Triangle);
  void pup(PUP::er& p) override { Shape::pup(p); }
};

class Square : public Shape {
 public:
  Square() = default;
  explicit Square(CkMigrateMessage* /*m*/) {}
  size_t number_of_sides() const noexcept final { return 4; }
  // clang-tidy: internal charm++ warnings
  WRAPPED_PUPable_decl_base_template(Shape,  // NOLINT
                                     Square);
  void pup(PUP::er& p) override { Shape::pup(p); }
};
#pragma GCC diagnostic pop

struct shape_of_nametag {
  using type = std::unique_ptr<Shape>;
};

namespace Tentacles {
struct TestGroup {
  using type = CProxy_TestGroupChare;
};

struct TestNodeGroup {
  using type = CProxy_TestNodeGroupChare;
};

struct Test {
  using type = CProxy_TestChare;
};

struct TestArray {
  using type = CProxy_TestArrayChare;
};
}  // namespace Tentacles

struct TestMetavariables {
  using const_global_cache_tag_list =
      typelist<name, age, height, shape_of_nametag>;
  using tentacle_list = typelist<Tentacles::TestGroup, Tentacles::TestNodeGroup,
                                 Tentacles::Test, Tentacles::TestArray>;
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.ConstGlobalCache", "[Unit][Parallel]") {
  tuples::TaggedTupleTypelist<
      typename TestMetavariables::const_global_cache_tag_list>
      const_data_to_be_cached("Nobody", 178, 2.2, std::make_unique<Square>());

  Parallel::ConstGlobalCache<TestMetavariables> cache(const_data_to_be_cached);

  tuples::TaggedTupleTypelist<typename TestMetavariables::tentacle_list>
      tentacles;

  auto& group_proxy = tuples::get<Tentacles::TestGroup>(tentacles);
  auto& node_group_proxy = tuples::get<Tentacles::TestNodeGroup>(tentacles);
  auto& proxy = tuples::get<Tentacles::Test>(tentacles);
  auto& array_proxy = tuples::get<Tentacles::TestArray>(tentacles);
  group_proxy = CProxy_TestGroupChare::ckNew();
  node_group_proxy = CProxy_TestNodeGroupChare::ckNew();
  proxy = CProxy_TestChare::ckNew(-1);
  array_proxy = CProxy_TestArrayChare::ckNew(40);

  CkCallback cb(CkCallback::ignore);
  cache.set_tentacles(tentacles, cb);

  auto& retrieved_proxy = cache.get_tentacle<Tentacles::Test>();
  auto& retrieved_array_proxy = cache.get_tentacle<Tentacles::TestArray>();

  CHECK(Parallel::my_proc() ==
        cache.get_tentacle<Tentacles::TestGroup>().ckLocalBranch()->my_proc());
  CHECK(Parallel::my_node() ==
        cache.get_tentacle<Tentacles::TestNodeGroup>()
            .ckLocalBranch()
            ->my_node());
  if (nullptr != retrieved_proxy.ckLocal()) {
    CHECK(-1 == retrieved_proxy.ckLocal()->my_id());
  }
  for (int i = 0; i < 40; ++i) {
    if (nullptr != retrieved_array_proxy[i].ckLocal()) {
      CHECK(i == retrieved_array_proxy[i].ckLocal()->my_index());
    }
  }
  CHECK("Nobody" == cache.get<name>());
  CHECK(178 == cache.get<age>());
  CHECK(2.2 == cache.get<height>());
  CHECK(4 == cache.get<shape_of_nametag>().number_of_sides());
}

Test_ConstGlobalCache::Test_ConstGlobalCache(CkArgMsg* msg) {
  std::set_terminate(
      []() { Parallel::abort("Called terminate. Aborting..."); });
  Parallel::printf("%s", info_from_build().c_str());
  enable_floating_point_exceptions();
  int result = Catch::Session().run(msg->argc, msg->argv);
  if (0 == result) {
    Parallel::exit();
  }
  Parallel::abort("A catch test has failed.");
}

/// \cond
// clang-tidy: possibly throwing constructor static storage
// clang-tidy: false positive: redundant declaration
PUP::able::PUP_ID Triangle::my_PUP_ID = 0;  // NOLINT
PUP::able::PUP_ID Square::my_PUP_ID = 0;    // NOLINT
/// \endcond

#include "src/Parallel/ConstGlobalCache.def.h"

#include "tests/Unit/Parallel/Test_ConstGlobalCache.def.h"
