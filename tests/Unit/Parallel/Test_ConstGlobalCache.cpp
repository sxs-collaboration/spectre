// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include "tests/Unit/TestingFramework.hpp"

void register_pupables();

#include "tests/Unit/Parallel/Test_ConstGlobalCache.hpp"

#include <algorithm>
#include <charm++.h>
#include <cstddef>
#include <exception>
#include <memory>
#include <string>

#include "AlgorithmArray.hpp"
#include "AlgorithmGroup.hpp"
#include "AlgorithmNodegroup.hpp"
#include "AlgorithmSingleton.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Parallel/Abort.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Exit.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

namespace Parallel {
namespace charmxx {
struct RegistrationHelper;
}  // namespace charmxx
}  // namespace Parallel

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

struct shape_of_nametag_base {};

struct shape_of_nametag : shape_of_nametag_base {
  using type = std::unique_ptr<Shape>;
};

template <class Metavariables>
struct SingletonParallelComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using const_global_cache_tag_list = tmpl::list<name, age, height>;
  using options = tmpl::list<>;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using initial_databox = db::DataBox<tmpl::list<>>;
};

template <class Metavariables>
struct ArrayParallelComponent {
  using chare_type = Parallel::Algorithms::Array;
  using const_global_cache_tag_list = tmpl::list<height, shape_of_nametag>;
  using array_index = int;
  using options = tmpl::list<>;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using initial_databox = db::DataBox<tmpl::list<>>;
};

template <class Metavariables>
struct GroupParallelComponent {
  using chare_type = Parallel::Algorithms::Group;
  using const_global_cache_tag_list = tmpl::list<name>;
  using options = tmpl::list<>;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using initial_databox = db::DataBox<tmpl::list<>>;
};

template <class Metavariables>
struct NodegroupParallelComponent {
  using chare_type = Parallel::Algorithms::Nodegroup;
  using const_global_cache_tag_list = tmpl::list<height>;
  using options = tmpl::list<>;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using initial_databox = db::DataBox<tmpl::list<>>;
};

struct TestMetavariables {
  using component_list =
      tmpl::list<SingletonParallelComponent<TestMetavariables>,
                 ArrayParallelComponent<TestMetavariables>,
                 GroupParallelComponent<TestMetavariables>,
                 NodegroupParallelComponent<TestMetavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.ConstGlobalCache", "[Unit][Parallel]") {
  {
    using tag_list =
        typename Parallel::ConstGlobalCache<TestMetavariables>::tag_list;
    static_assert(
        cpp17::is_same_v<tag_list,
                         tmpl::list<name, age, height, shape_of_nametag>>,
        "Wrong tag_list in ConstGlobalCache test");

    tuples::tagged_tuple_from_typelist<tag_list> const_data_to_be_cached(
        "Nobody", 178, 2.2, std::make_unique<Square>());
    Parallel::ConstGlobalCache<TestMetavariables> cache(
        std::move(const_data_to_be_cached));
    CHECK("Nobody" == Parallel::get<name>(cache));
    CHECK(178 == Parallel::get<age>(cache));
    CHECK(2.2 == Parallel::get<height>(cache));
    CHECK(4 == Parallel::get<shape_of_nametag>(cache).number_of_sides());
    CHECK(4 == Parallel::get<shape_of_nametag_base>(cache).number_of_sides());
  }

  {
    using tag_list =
        typename Parallel::ConstGlobalCache<TestMetavariables>::tag_list;
    static_assert(
        cpp17::is_same_v<tag_list,
                         tmpl::list<name, age, height, shape_of_nametag>>,
        "Wrong tag_list in ConstGlobalCache test");

    tuples::tagged_tuple_from_typelist<tag_list> const_data_to_be_cached(
        "Nobody", 178, 2.2, std::make_unique<Square>());

    Parallel::CProxy_ConstGlobalCache<TestMetavariables>
        const_global_cache_proxy =
            Parallel::CProxy_ConstGlobalCache<TestMetavariables>::ckNew(
                const_data_to_be_cached);
    const auto& local_cache = *const_global_cache_proxy.ckLocalBranch();
    CHECK("Nobody" == Parallel::get<name>(local_cache));
    CHECK(178 == Parallel::get<age>(local_cache));
    CHECK(2.2 == Parallel::get<height>(local_cache));
    CHECK(4 == Parallel::get<shape_of_nametag>(local_cache).number_of_sides());
    CHECK(4 ==
          Parallel::get<shape_of_nametag_base>(local_cache).number_of_sides());
  }
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

void register_pupables() {
  PUPable_reg(Triangle);
  PUPable_reg(Square);
}

/// \cond
// clang-tidy: possibly throwing constructor static storage
// clang-tidy: false positive: redundant declaration
PUP::able::PUP_ID Triangle::my_PUP_ID = 0;  // NOLINT
PUP::able::PUP_ID Square::my_PUP_ID = 0;    // NOLINT
/// \endcond

#include "src/Parallel/ConstGlobalCache.def.h"  // IWYU pragma: keep

#include "tests/Unit/Parallel/Test_ConstGlobalCache.def.h"  // IWYU pragma: keep

namespace Parallel {
namespace charmxx {
/// \cond
std::unique_ptr<RegistrationHelper>* charm_register_list = nullptr;
size_t charm_register_list_capacity = 0;
size_t charm_register_list_size = 0;
/// \endcond
}  // namespace charmxx
}  // namespace Parallel
