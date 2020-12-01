// Distributed under the MIT License.
// See LICENSE.txt for details.

// Need CATCH_CONFIG_RUNNER to avoid linking errors with Catch2
#define CATCH_CONFIG_RUNNER

#include "Framework/TestingFramework.hpp"

#include "Parallel/Test_GlobalCache.hpp"

#include <charm++.h>
#include <cstddef>
#include <functional>
#include <optional>
#include <pup.h>
#include <string>
#include <tuple>

#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/Algorithms/AlgorithmGroup.hpp"
#include "Parallel/Algorithms/AlgorithmNodegroup.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/Exit.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

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

struct weight {
  using type = double;
};

struct email {
  using type = std::string;
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

class Animal : public PUP::able {
 public:
  Animal() = default;
  virtual size_t number_of_legs() const = 0;
  virtual void set_number_of_legs(size_t) = 0;
  // clang-tidy: internal charm++ warnings
  WRAPPED_PUPable_abstract(Animal);  // NOLINT
};

class Arthropod : public Animal {
 public:
  Arthropod() = default;
  explicit Arthropod(const size_t num_legs) : number_of_legs_(num_legs){};
  explicit Arthropod(CkMigrateMessage* /*m*/) {}
  size_t number_of_legs() const noexcept final { return number_of_legs_; }
  void set_number_of_legs(const size_t num_legs) noexcept final {
    number_of_legs_ = num_legs;
  }
  // clang-tidy: internal charm++ warnings
  WRAPPED_PUPable_decl_base_template(Animal,  // NOLINT
                                     Arthropod);

  void pup(PUP::er& p) override {
    Animal::pup(p);
    p | number_of_legs_;
  }

 private:
  size_t number_of_legs_{0};
};
#pragma GCC diagnostic pop

struct shape_of_nametag_base {};

struct shape_of_nametag : shape_of_nametag_base {
  using type = std::unique_ptr<Shape>;
};

struct animal_base {};

struct animal : animal_base {
  using type = std::unique_ptr<Animal>;
};

template <typename T>
struct modify_value {
  static void apply(const gsl::not_null<T*> value,
                    const T& new_value) noexcept {
    *value = new_value;
  }
};

struct modify_number_of_legs {
  static void apply(const gsl::not_null<std::unique_ptr<Animal>*> animal_local,
                    const size_t num_legs) noexcept {
    (*animal_local)->set_number_of_legs(num_legs);
  }
};

template <class Metavariables>
struct SingletonParallelComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using const_global_cache_tags = tmpl::list<name, age, height>;
  using mutable_global_cache_tags = tmpl::list<weight, animal>;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;
};

template <class Metavariables>
struct ArrayParallelComponent {
  using chare_type = Parallel::Algorithms::Array;
  using const_global_cache_tags = tmpl::list<height, shape_of_nametag>;
  using mutable_global_cache_tags = tmpl::list<email, weight>;
  using array_index = int;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;
};

template <class Metavariables>
struct GroupParallelComponent {
  using chare_type = Parallel::Algorithms::Group;
  using const_global_cache_tags = tmpl::list<name>;
  using mutable_global_cache_tags = tmpl::list<email>;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;
};

template <class Metavariables>
struct NodegroupParallelComponent {
  using chare_type = Parallel::Algorithms::Nodegroup;
  using const_global_cache_tags = tmpl::list<height>;
  using mutable_global_cache_tags = tmpl::list<animal>;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;
};

struct TestMetavariables {
  using component_list =
      tmpl::list<SingletonParallelComponent<TestMetavariables>,
                 ArrayParallelComponent<TestMetavariables>,
                 GroupParallelComponent<TestMetavariables>,
                 NodegroupParallelComponent<TestMetavariables>>;
  enum class Phase { Testing, Exit };
};

}  // namespace

template <typename Metavariables>
void TestArrayChare<Metavariables>::run_test_one() noexcept {
  // Test that the values are what we think they should be.
  auto& local_cache = *global_cache_proxy_.ckLocalBranch();
  SPECTRE_PARALLEL_REQUIRE("Nobody" == Parallel::get<name>(local_cache));
  SPECTRE_PARALLEL_REQUIRE(178 == Parallel::get<age>(local_cache));
  SPECTRE_PARALLEL_REQUIRE(2.2 == Parallel::get<height>(local_cache));
  SPECTRE_PARALLEL_REQUIRE(
      4 == Parallel::get<shape_of_nametag>(local_cache).number_of_sides());
  SPECTRE_PARALLEL_REQUIRE(
      4 == Parallel::get<shape_of_nametag_base>(local_cache).number_of_sides());
  SPECTRE_PARALLEL_REQUIRE(160 == Parallel::get<weight>(local_cache));
  SPECTRE_PARALLEL_REQUIRE("joe@somewhere.com" ==
                           Parallel::get<email>(local_cache));
  SPECTRE_PARALLEL_REQUIRE(6 ==
                           Parallel::get<animal>(local_cache).number_of_legs());
  SPECTRE_PARALLEL_REQUIRE(
      6 == Parallel::get<animal_base>(local_cache).number_of_legs());

  const auto local_cache_from_proxy =
      local_cache.get_this_proxy().ckLocalBranch();
  SPECTRE_PARALLEL_REQUIRE(local_cache_from_proxy ==
                           global_cache_proxy_.ckLocalBranch());

  // Mutate the weight to 150.
  Parallel::mutate<weight, modify_value<double>>(global_cache_proxy_, 150.0);
  run_test_two();
}

template <typename Metavariables>
void TestArrayChare<Metavariables>::run_test_two() noexcept {
  // Move on when the weight is 150.
  auto callback =
      CkCallback(CkIndex_TestArrayChare<Metavariables>::run_test_two(),
                 this->thisProxy[this->thisIndex]);
  if (Parallel::mutable_cache_item_is_ready<weight>(
          *global_cache_proxy_.ckLocalBranch(),
          [&callback](const double& weight_l) -> std::optional<CkCallback> {
            return weight_l == 150 ? std::optional<CkCallback>{}
                                   : std::optional<CkCallback>(callback);
          })) {
    auto& local_cache = *global_cache_proxy_.ckLocalBranch();
    SPECTRE_PARALLEL_REQUIRE(150 == Parallel::get<weight>(local_cache));

    // Now the weight is 150, so mutate the email.
    Parallel::mutate<email, modify_value<std::string>>(
        global_cache_proxy_, std::string("albert@einstein.de"));
    // ... and make the arthropod into a lobster.
    Parallel::mutate<animal, modify_number_of_legs>(global_cache_proxy_, 10_st);
    run_test_three();
  }
}

template <typename Metavariables>
void TestArrayChare<Metavariables>::run_test_three() noexcept {
  // Move on when the email is Albert's.
  auto callback =
      CkCallback(CkIndex_TestArrayChare<Metavariables>::run_test_three(),
                 this->thisProxy[this->thisIndex]);
  if (Parallel::mutable_cache_item_is_ready<email>(
          *global_cache_proxy_.ckLocalBranch(),
          [&callback](const std::string& email_l) -> std::optional<CkCallback> {
            return email_l == "albert@einstein.de"
                       ? std::optional<CkCallback>{}
                       : std::optional<CkCallback>(callback);
          })) {
    auto& local_cache = *global_cache_proxy_.ckLocalBranch();
    SPECTRE_PARALLEL_REQUIRE("albert@einstein.de" ==
                             Parallel::get<email>(local_cache));

    // Now make the arthropod into a spider.
    Parallel::mutate<animal, modify_number_of_legs>(global_cache_proxy_, 8_st);
    run_test_four();
  }
}

template <typename Metavariables>
void TestArrayChare<Metavariables>::run_test_four() noexcept {
  // Move on when the animal has 8 legs.
  auto callback =
      CkCallback(CkIndex_TestArrayChare<Metavariables>::run_test_four(),
                 this->thisProxy[this->thisIndex]);
  if (Parallel::mutable_cache_item_is_ready<animal>(
          *global_cache_proxy_.ckLocalBranch(),
          [&callback](const Animal& animal_l) -> std::optional<CkCallback> {
            return animal_l.number_of_legs() == 8
                       ? std::optional<CkCallback>{}
                       : std::optional<CkCallback>(callback);
          })) {
    auto& local_cache = *global_cache_proxy_.ckLocalBranch();
    SPECTRE_PARALLEL_REQUIRE(
        8 == Parallel::get<animal>(local_cache).number_of_legs());

    // Make the arthropod into a Scutigera coleoptrata.
    Parallel::mutate<animal_base, modify_number_of_legs>(global_cache_proxy_,
                                                         30_st);

    run_test_five();
  }
}

template <typename Metavariables>
void TestArrayChare<Metavariables>::run_test_five() noexcept {
  // Move on when the animal has 30 legs.
  auto callback =
      CkCallback(CkIndex_TestArrayChare<Metavariables>::run_test_five(),
                 this->thisProxy[this->thisIndex]);
  if (Parallel::mutable_cache_item_is_ready<animal_base>(
          *global_cache_proxy_.ckLocalBranch(),
          [&callback](const Animal& animal_l) -> std::optional<CkCallback> {
            return animal_l.number_of_legs() == 30
                       ? std::optional<CkCallback>{}
                       : std::optional<CkCallback>(callback);
          })) {
    auto& local_cache = *global_cache_proxy_.ckLocalBranch();
    SPECTRE_PARALLEL_REQUIRE(
        30 == Parallel::get<animal_base>(local_cache).number_of_legs());
    main_proxy_.exit_if_done(this->thisIndex);
  }
}

// run_single_core_test constructs and tests GlobalCache without
// using proxies or parallelism.  run_single_core_test tests constructors
// and member functions that are used in the action testing framework.
template <typename Metavariables>
void Test_GlobalCache<Metavariables>::run_single_core_test() noexcept {
  using const_tag_list =
      typename Parallel::get_const_global_cache_tags<TestMetavariables>;
  static_assert(std::is_same_v<const_tag_list,
                               tmpl::list<name, age, height, shape_of_nametag>>,
                "Wrong const_tag_list in GlobalCache test");

  using mutable_tag_list =
      typename Parallel::get_mutable_global_cache_tags<TestMetavariables>;
  static_assert(
      std::is_same_v<mutable_tag_list, tmpl::list<weight, animal, email>>,
      "Wrong mutable_tag_list in GlobalCache test");

  tuples::tagged_tuple_from_typelist<const_tag_list> const_data_to_be_cached(
      "Nobody", 178, 2.2, std::make_unique<Square>());
  tuples::tagged_tuple_from_typelist<mutable_tag_list>
      mutable_data_to_be_cached(160, std::make_unique<Arthropod>(6),
                                "joe@somewhere.com");

  Parallel::MutableGlobalCache<TestMetavariables> mutable_cache(
      std::move(mutable_data_to_be_cached));
  Parallel::GlobalCache<TestMetavariables> cache(
      std::move(const_data_to_be_cached), &mutable_cache);
  SPECTRE_PARALLEL_REQUIRE("Nobody" == Parallel::get<name>(cache));
  SPECTRE_PARALLEL_REQUIRE(178 == Parallel::get<age>(cache));
  SPECTRE_PARALLEL_REQUIRE(2.2 == Parallel::get<height>(cache));
  SPECTRE_PARALLEL_REQUIRE(
      4 == Parallel::get<shape_of_nametag>(cache).number_of_sides());
  SPECTRE_PARALLEL_REQUIRE(
      4 == Parallel::get<shape_of_nametag_base>(cache).number_of_sides());
  SPECTRE_PARALLEL_REQUIRE(160 == Parallel::get<weight>(cache));
  SPECTRE_PARALLEL_REQUIRE("joe@somewhere.com" == Parallel::get<email>(cache));
  SPECTRE_PARALLEL_REQUIRE(6 == Parallel::get<animal>(cache).number_of_legs());
  SPECTRE_PARALLEL_REQUIRE(6 ==
                           Parallel::get<animal_base>(cache).number_of_legs());

  // Check that we can modify the non-const items.
  Parallel::mutate<weight, modify_value<double>>(cache, 150.0);
  Parallel::mutate<email, modify_value<std::string>>(
      cache, std::string("nobody@nowhere.com"));
  SPECTRE_PARALLEL_REQUIRE(150 == Parallel::get<weight>(cache));
  SPECTRE_PARALLEL_REQUIRE("nobody@nowhere.com" == Parallel::get<email>(cache));
  Parallel::mutate<email, modify_value<std::string>>(
      cache, std::string("isaac@newton.com"));
  SPECTRE_PARALLEL_REQUIRE("isaac@newton.com" == Parallel::get<email>(cache));
  // Make the arthropod into a spider.
  Parallel::mutate<animal, modify_number_of_legs>(cache, 8_st);
  SPECTRE_PARALLEL_REQUIRE(8 == Parallel::get<animal>(cache).number_of_legs());
  SPECTRE_PARALLEL_REQUIRE(8 ==
                           Parallel::get<animal_base>(cache).number_of_legs());
  // Make the arthropod into a Scutigera coleoptrata.
  Parallel::mutate<animal_base, modify_number_of_legs>(cache, 30_st);
  SPECTRE_PARALLEL_REQUIRE(30 == Parallel::get<animal>(cache).number_of_legs());
  SPECTRE_PARALLEL_REQUIRE(30 ==
                           Parallel::get<animal_base>(cache).number_of_legs());
}

template <typename Metavariables>
Test_GlobalCache<Metavariables>::Test_GlobalCache(CkArgMsg*
                                                  /*msg*/) noexcept {
  // Register the pup functions.
  Parallel::register_classes_in_list<tmpl::list<Triangle, Square, Arthropod>>();

  // Call the single core test before doing anything else.
  run_single_core_test();

  // Initialize number of elements
  num_elements_ = 4;

  // Create GlobalCache proxies.
  using mutable_tag_list =
      typename Parallel::get_mutable_global_cache_tags<TestMetavariables>;
  static_assert(
      std::is_same_v<mutable_tag_list, tmpl::list<weight, animal, email>>,
      "Wrong mutable_tag_list in GlobalCache test");
  // Arthropod begins as an insect.
  tuples::tagged_tuple_from_typelist<mutable_tag_list>
      mutable_data_to_be_cached(160, std::make_unique<Arthropod>(6),
                                "joe@somewhere.com");
  mutable_global_cache_proxy_ =
      Parallel::CProxy_MutableGlobalCache<TestMetavariables>::ckNew(
          mutable_data_to_be_cached);

  // global_cache_proxy_ depends on mutable_global_cache_proxy_.
  CkEntryOptions mutable_global_cache_dependency;
  mutable_global_cache_dependency.setGroupDepID(
      mutable_global_cache_proxy_.ckGetGroupID());

  using const_tag_list =
      typename Parallel::get_const_global_cache_tags<TestMetavariables>;
  static_assert(std::is_same_v<const_tag_list,
                               tmpl::list<name, age, height, shape_of_nametag>>,
                "Wrong const_tag_list in GlobalCache test");
  tuples::tagged_tuple_from_typelist<const_tag_list> const_data_to_be_cached(
      "Nobody", 178, 2.2, std::make_unique<Square>());
  global_cache_proxy_ = Parallel::CProxy_GlobalCache<TestMetavariables>::ckNew(
      const_data_to_be_cached, mutable_global_cache_proxy_,
      &mutable_global_cache_dependency);

  CkEntryOptions global_cache_dependency;
  global_cache_dependency.setGroupDepID(global_cache_proxy_.ckGetGroupID());

  // Create array
  CProxy_TestArrayChare<Metavariables> array_proxy =
      CProxy_TestArrayChare<Metavariables>::ckNew(
          this->thisProxy, global_cache_proxy_, num_elements_);

  array_proxy.run_test_one();
}

template <typename Metavariables>
void Test_GlobalCache<Metavariables>::exit_if_done(int index) noexcept {
  elements_that_are_finished_.insert(index);
  if (elements_that_are_finished_.size() >= num_elements_) {
    Parallel::exit();
  }
}

// --------- registration stuff below -------

/// \cond
// clang-tidy: possibly throwing constructor static storage
// clang-tidy: false positive: redundant declaration
PUP::able::PUP_ID Triangle::my_PUP_ID = 0;   // NOLINT
PUP::able::PUP_ID Square::my_PUP_ID = 0;     // NOLINT
PUP::able::PUP_ID Arthropod::my_PUP_ID = 0;  // NOLINT
/// \endcond

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};

using charmxx_main_component = Test_GlobalCache<TestMetavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
