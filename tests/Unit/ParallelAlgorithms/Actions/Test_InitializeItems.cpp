// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct TagOne : db::SimpleTag {
  using type = int;
};

struct TagTwo : db::SimpleTag {
  using type = std::string;
};

struct TagThree : db::SimpleTag {
  using type = double;
};

struct Two : db::SimpleTag {
  using type = double;
};

struct Three : db::SimpleTag {
  using type = double;
};

struct Five : db::SimpleTag {
  using type = double;
};

struct Ten : db::SimpleTag {
  using type = double;
};

struct FiveCompute : Five, db::ComputeTag {
  using return_type = double;
  using base = Five;
  using argument_tags = tmpl::list<Two, Three>;
  static void function(const gsl::not_null<double*> result, const double two,
                       const double three) {
    *result = two + three;
  }
};

struct Duplicate : db::SimpleTag {
  using type = int;
};

struct InitializeThree {
  using const_global_cache_tags = tmpl::list<TagOne, Duplicate>;
  using mutable_global_cache_tags = tmpl::list<TagTwo, Duplicate>;
  using simple_tags_from_options = tmpl::list<>;
  using simple_tags = tmpl::list<Three, Duplicate>;
  using compute_tags = tmpl::list<>;
  using return_tags = tmpl::list<Three>;
  using argument_tags = tmpl::list<>;

  static void apply(const gsl::not_null<double*> three) { *three = 3.; }
};

struct InitializeTwo {
  using const_global_cache_tags = tmpl::list<TagThree, Duplicate>;
  using mutable_global_cache_tags = tmpl::list<Duplicate>;
  using simple_tags_from_options = tmpl::list<Two>;
  using simple_tags = tmpl::list<Duplicate>;
  using compute_tags = tmpl::list<FiveCompute>;
  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<>;
  static void apply() {}
};

struct InitializeTen {
  using const_global_cache_tags = tmpl::list<Duplicate>;
  using mutable_global_cache_tags = tmpl::list<Duplicate>;
  using simple_tags_from_options = tmpl::list<Two>;
  using simple_tags = tmpl::list<Ten, Duplicate>;
  using compute_tags = tmpl::list<FiveCompute>;
  using return_tags = tmpl::list<Ten>;
  using argument_tags = tmpl::list<Two, Five>;

  static void apply(const gsl::not_null<double*> ten, const double two,
                    const double five) {
    *ten = two * five;
  }
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;

  using initialization_action =
      Initialization::Actions::InitializeItems<InitializeTwo, InitializeThree,
                                               InitializeTen>;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<Parallel::Phase::Initialization,
                                        tmpl::list<initialization_action>>>;

  static_assert(std::is_same_v<
                tmpl::count_if<initialization_action::const_global_cache_tags,
                               std::is_same<tmpl::_1, Duplicate>>,
                tmpl::size_t<1>>);
  static_assert(std::is_same_v<
                tmpl::count_if<initialization_action::mutable_global_cache_tags,
                               std::is_same<tmpl::_1, Duplicate>>,
                tmpl::size_t<1>>);
  static_assert(std::is_same_v<
                tmpl::count_if<initialization_action::simple_tags_from_options,
                               std::is_same<tmpl::_1, Two>>,
                tmpl::size_t<1>>);
  static_assert(
      std::is_same_v<tmpl::count_if<initialization_action::simple_tags,
                                    std::is_same<tmpl::_1, Duplicate>>,
                     tmpl::size_t<1>>);
  static_assert(
      std::is_same_v<tmpl::count_if<initialization_action::compute_tags,
                                    std::is_same<tmpl::_1, FiveCompute>>,
                     tmpl::size_t<1>>);

  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;
};

struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
};

}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Actions.InitializeItems",

                  "[Unit][ParallelAlgorithms]") {
  using component = Component<Metavariables>;

  tuples::TaggedTuple<TagOne, TagThree, Duplicate> const_cache_items{7, -4.,
                                                                     8.};
  tuples::TaggedTuple<TagTwo, Duplicate> mutable_cache_items{"bla", 9.};

  ActionTesting::MockRuntimeSystem<Metavariables> runner{const_cache_items,
                                                         mutable_cache_items};
  ActionTesting::emplace_array_component<component>(make_not_null(&runner), {0},
                                                    {0}, 0, 2.);

  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  for (size_t i = 0; i < 1; ++i) {
    runner.template next_action<component>(0);
  }

  CHECK(ActionTesting::tag_is_retrievable<component, TagOne>(runner, 0));
  CHECK(ActionTesting::tag_is_retrievable<component, TagThree>(runner, 0));
  CHECK(ActionTesting::get_databox_tag<component, Two>(runner, 0) == 2.);
  CHECK(ActionTesting::get_databox_tag<component, Three>(runner, 0) == 3.);
  CHECK(ActionTesting::get_databox_tag<component, Ten>(runner, 0) == 10.);
  CHECK(ActionTesting::get_databox_tag<component, Five>(runner, 0) == 5.);
}
