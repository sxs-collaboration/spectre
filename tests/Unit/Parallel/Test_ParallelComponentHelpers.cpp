// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <vector>

#include "Options/ParseOptions.hpp"
#include "Parallel/CreateFromOptions.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"  // IWYU pragma: keep
#include "Utilities/NoSuchType.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

namespace {
struct Tag0 {};
struct Tag1 {};
struct Tag2 {};
struct Tag3 {};
struct Tag4 {};
struct Tag5 {};
struct Tag6 {};
struct Tag7 {};

struct Action0 {
  using inbox_tags = tmpl::list<Tag0, Tag1>;
};
struct Action1 {};
struct Action2 {
  using inbox_tags = tmpl::list<Tag1, Tag2, Tag3>;
  using const_global_cache_tags = tmpl::list<Tag6>;
};

struct InitTag0 {};
struct InitTag1 {};
struct InitTag2 {};
struct InitTag3 {};
struct InitTag4 {};
struct InitTag5 {};
struct InitTag6 {};
struct InitTag7 {};

struct InitAction0 {
  using initialization_tags = tmpl::list<InitTag0>;
};
struct InitAction1 {};
struct InitAction2 {
  using initialization_tags = tmpl::list<InitTag1, InitTag2>;
};
struct InitAction3 {
  using initialization_tags = tmpl::list<InitTag0, InitTag1, InitTag3>;
};

enum class Phase { Initialization, Execute, Exit };

struct ComponentInit {
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Phase, Phase::Initialization,
      tmpl::list<InitAction0, InitAction1, InitAction2>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;
};

struct ComponentExecute {
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<Phase, Phase::Execute,
                                        tmpl::list<Action0, Action1>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;
  using const_global_cache_tags = tmpl::list<Tag2, Tag4>;
};

struct ComponentInitAndExecute {
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Phase, Phase::Initialization,
                             tmpl::list<InitAction3>>,
      Parallel::PhaseActions<Phase, Phase::Execute,
                             tmpl::list<InitAction2, Action0, Action2>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;
};

struct ComponentInitWithAllocate {
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Phase, Phase::Initialization,
      tmpl::list<InitAction0, InitAction1, InitAction2>>>;
  using array_allocation_tags = tmpl::list<InitTag4, InitTag5>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>,
      array_allocation_tags>;
  using const_global_cache_tags = tmpl::list<Tag1, Tag5, Tag7>;
};

struct ComponentExecuteWithAllocate {
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<Phase, Phase::Execute,
                                        tmpl::list<Action0, Action1>>>;
  using array_allocation_tags = tmpl::list<InitTag6, InitTag7>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>,
      array_allocation_tags>;
};

struct ComponentInitAndExecuteWithAllocate {
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Phase, Phase::Initialization,
                             tmpl::list<InitAction3>>,
      Parallel::PhaseActions<Phase, Phase::Execute,
                             tmpl::list<InitAction2, Action0, Action2>>>;
  using array_allocation_tags = tmpl::list<InitTag3, InitTag4>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>,
      array_allocation_tags>;
  using const_global_cache_tags = tmpl::list<Tag2, Tag4>;
};

struct Metavariables0 {
  using component_list = tmpl::list<ComponentInit>;
};

struct Metavariables1 {
  using const_global_cache_tags = tmpl::list<Tag0, Tag4>;
  using component_list = tmpl::list<>;
};

struct Metavariables2 {
  using component_list = tmpl::list<ComponentInitWithAllocate>;
};

struct Metavariables3 {
  using const_global_cache_tags = tmpl::list<Tag0, Tag4>;
  using component_list = tmpl::list<ComponentInitWithAllocate>;
};

struct Metavariables4 {
  using component_list = tmpl::list<ComponentInitAndExecute>;
};

struct Metavariables5 {
  using const_global_cache_tags = tmpl::list<Tag0, Tag4>;
  using component_list = tmpl::list<ComponentInitAndExecute>;
};

struct Metavariables6 {
  using component_list = tmpl::list<ComponentInitAndExecuteWithAllocate>;
};

struct Metavariables7 {
  using const_global_cache_tags = tmpl::list<Tag0, Tag4>;
  using component_list = tmpl::list<ComponentInitAndExecuteWithAllocate>;
};

static_assert(
    std::is_same_v<
        Parallel::get_inbox_tags<tmpl::list<Action0, Action1, Action2>>,
        tmpl::list<Tag0, Tag1, Tag2, Tag3>>,
    "Failed testing get_inbox_tags");

static_assert(
    std::is_same_v<Parallel::get_const_global_cache_tags<Metavariables0>,
                   tmpl::list<>>,
    "Failed testing get_const_global_cache_tags");

static_assert(
    std::is_same_v<Parallel::get_const_global_cache_tags<Metavariables1>,
                   tmpl::list<Tag0, Tag4>>,
    "Failed testing get_const_global_cache_tags");

static_assert(
    std::is_same_v<Parallel::get_const_global_cache_tags<Metavariables2>,
                   tmpl::list<Tag1, Tag5, Tag7>>,
    "Failed testing get_const_global_cache_tags");

static_assert(
    std::is_same_v<Parallel::get_const_global_cache_tags<Metavariables3>,
                   tmpl::list<Tag0, Tag4, Tag1, Tag5, Tag7>>,
    "Failed testing get_const_global_cache_tags");

static_assert(
    std::is_same_v<Parallel::get_const_global_cache_tags<Metavariables4>,
                   tmpl::list<Tag6>>,
    "Failed testing get_const_global_cache_tags");

static_assert(
    std::is_same_v<Parallel::get_const_global_cache_tags<Metavariables5>,
                   tmpl::list<Tag0, Tag4, Tag6>>,
    "Failed testing get_const_global_cache_tags");

static_assert(
    std::is_same_v<Parallel::get_const_global_cache_tags<Metavariables6>,
                   tmpl::list<Tag2, Tag4, Tag6>>,
    "Failed testing get_const_global_cache_tags");

static_assert(
    std::is_same_v<Parallel::get_const_global_cache_tags<Metavariables7>,
                   tmpl::list<Tag0, Tag4, Tag2, Tag6>>,
    "Failed testing get_const_global_cache_tags");

static_assert(std::is_same_v<Parallel::get_initialization_actions_list<
                                 ComponentInit::phase_dependent_action_list>,
                             tmpl::list<InitAction0, InitAction1, InitAction2>>,
              "Failed testing get_intialization_actions_list");

static_assert(std::is_same_v<Parallel::get_initialization_actions_list<
                                 ComponentExecute::phase_dependent_action_list>,
                             tmpl::list<>>,
              "Failed testing get_intialization_actions_list");

static_assert(
    std::is_same_v<Parallel::get_initialization_actions_list<
                       ComponentInitAndExecute::phase_dependent_action_list>,
                   tmpl::list<InitAction3>>,
    "Failed testing get_intialization_actions_list");

static_assert(std::is_same_v<ComponentInit::initialization_tags,
                             tmpl::list<InitTag0, InitTag1, InitTag2>>,
              "Failed testing get_initialization_tags");

static_assert(
    std::is_same_v<ComponentExecute::initialization_tags, tmpl::list<>>,
    "Failed testing get_initialization_tags");

static_assert(std::is_same_v<ComponentInitAndExecute::initialization_tags,
                             tmpl::list<InitTag0, InitTag1, InitTag3>>,
              "Failed testing get_initialization_tags");

static_assert(std::is_same_v<
                  ComponentInitWithAllocate::initialization_tags,
                  tmpl::list<InitTag4, InitTag5, InitTag0, InitTag1, InitTag2>>,
              "Failed testing get_initialization_tags");

static_assert(std::is_same_v<ComponentExecuteWithAllocate::initialization_tags,
                             tmpl::list<InitTag6, InitTag7>>,
              "Failed testing get_initialization_tags");

static_assert(
    std::is_same_v<ComponentInitAndExecuteWithAllocate::initialization_tags,
                   tmpl::list<InitTag3, InitTag4, InitTag0, InitTag1>>,
    "Failed testing get_initialization_tags");

namespace OptionTags {
struct Yards {
  using type = double;
  static constexpr Options::String help = {"halp_yards"};
};
struct Dim {
  using type = size_t;
  static constexpr Options::String help = {"halp_size"};
};
struct Greeting {
  using type = std::string;
  static constexpr Options::String help = {"halp_greeting"};
};
struct Name {
  using type = std::string;
  static constexpr Options::String help = {"halp_name"};
};
}  // namespace OptionTags

namespace Initialization {
struct MetavariablesGreeting {};

namespace Tags {
struct Yards {
  using option_tags = tmpl::list<OptionTags::Yards>;
  using type = double;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double yards) noexcept {
    return yards;
  }
};
struct Feet {
  using option_tags = tmpl::list<OptionTags::Yards>;
  using type = double;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double yards) noexcept {
    return 3.0 * yards;
  }
};
struct Sides {
  using option_tags = tmpl::list<OptionTags::Yards, OptionTags::Dim>;
  using type = std::vector<double>;

  static constexpr bool pass_metavariables = false;
  static std::vector<double> create_from_options(const double yards,
                                                 const size_t dim) noexcept {
    return std::vector<double>(dim, yards);
  }
};
struct FullGreeting {
  using type = std::string;

  static constexpr bool pass_metavariables = true;
  template <typename Metavariables>
  using option_tags = tmpl::list<OptionTags::Greeting, OptionTags::Name>;
  template <typename Metavariables>
  static std::string create_from_options(const std::string& greeting,
                                         const std::string& name) noexcept {
    if (std::is_same<Metavariables, MetavariablesGreeting>::value) {
      return "A special " +  greeting + ' ' + name;
    } else {
      return greeting + ' ' + name;
    }
  }
};
}  // namespace Tags
}  // namespace Initialization

using initialization_tags_0 = tmpl::list<>;

using initialization_tags_1 =
    tmpl::list<Initialization::Tags::Yards, Initialization::Tags::Feet,
               Initialization::Tags::Sides>;

using initialization_tags_2 =
    tmpl::list<Initialization::Tags::Yards, Initialization::Tags::Feet,
               Initialization::Tags::FullGreeting>;

static_assert(
    std::is_same_v<Parallel::get_option_tags<initialization_tags_0, NoSuchType>,
                   tmpl::list<>>,
    "Failed testing get_option_tags");

static_assert(
    std::is_same_v<Parallel::get_option_tags<initialization_tags_1, NoSuchType>,
                   tmpl::list<OptionTags::Yards, OptionTags::Dim>>,
    "Failed testing get_option_tags");

static_assert(
    std::is_same_v<
        Parallel::get_option_tags<initialization_tags_2, NoSuchType>,
        tmpl::list<OptionTags::Yards, OptionTags::Greeting, OptionTags::Name>>,
    "Failed testing get_option_tags");

using all_option_tags = tmpl::list<OptionTags::Yards, OptionTags::Dim,
                                   OptionTags::Greeting, OptionTags::Name>;

template <typename Metavariables, typename... InitializationTags>
void check_initialization_items(
    const Options::Parser<all_option_tags>& all_options,
    const tuples::TaggedTuple<InitializationTags...>& expected_items) {
  using initialization_tags = tmpl::list<InitializationTags...>;
  using option_tags =
      Parallel::get_option_tags<initialization_tags, Metavariables>;
  const auto options = all_options.apply<option_tags>([](
      auto... args) noexcept {
    return tuples::tagged_tuple_from_typelist<option_tags>(std::move(args)...);
  });
  CHECK(Parallel::create_from_options<Metavariables>(
            options, initialization_tags{}) == expected_items);
}
}  // namespace


SPECTRE_TEST_CASE("Unit.Parallel.ComponentHelpers", "[Unit][Parallel]") {
  Options::Parser<all_option_tags> all_options("");
  all_options.parse(
      "Yards: 2.0\n"
      "Dim: 3\n"
      "Greeting: Hello\n"
      "Name: World\n");
  check_initialization_items<NoSuchType>(
      all_options, tuples::TaggedTuple<Initialization::Tags::Yards,
                                       Initialization::Tags::Feet,
                                       Initialization::Tags::Sides>{
                       2.0, 6.0, std::vector<double>{2.0, 2.0, 2.0}});
  check_initialization_items<NoSuchType>(
      all_options, tuples::TaggedTuple<Initialization::Tags::Yards,
                                       Initialization::Tags::Feet,
                                       Initialization::Tags::FullGreeting>{
                       2.0, 6.0, "Hello World"});
  check_initialization_items<Initialization::MetavariablesGreeting>(
      all_options, tuples::TaggedTuple<Initialization::Tags::Yards,
                                       Initialization::Tags::Feet,
                                       Initialization::Tags::FullGreeting>{
                       2.0, 6.0, "A special Hello World"});
}
