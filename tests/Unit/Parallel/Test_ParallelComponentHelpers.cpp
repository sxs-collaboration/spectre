// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <vector>

#include "Options/ParseOptions.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

namespace {
struct Tag0 {};
struct Tag1 {};
struct Tag2 {};
struct Tag3 {};

struct Action0 {
  using inbox_tags = tmpl::list<Tag0, Tag1>;
};
struct Action1 {};
struct Action2 {
  using inbox_tags = tmpl::list<Tag1, Tag2, Tag3>;
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
  using allocation_tags = tmpl::list<InitTag4, InitTag5>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>,
      allocation_tags>;
};

struct ComponentExecuteWithAllocate {
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<Phase, Phase::Execute,
                                        tmpl::list<Action0, Action1>>>;
  using allocation_tags = tmpl::list<InitTag6, InitTag7>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>,
      allocation_tags>;
};

struct ComponentInitAndExecuteWithAllocate {
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Phase, Phase::Initialization,
                             tmpl::list<InitAction3>>,
      Parallel::PhaseActions<Phase, Phase::Execute,
                             tmpl::list<InitAction2, Action0, Action2>>>;
  using allocation_tags = tmpl::list<InitTag3, InitTag4>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>,
      allocation_tags>;
};

static_assert(cpp17::is_same_v<Parallel::get_inbox_tags_from_action<Action2>,
                               tmpl::list<Tag1, Tag2, Tag3>>,
              "Failed testing get_inbox_tags_from_action");

static_assert(
    cpp17::is_same_v<
        Parallel::get_inbox_tags<tmpl::list<Action0, Action1, Action2>>,
        tmpl::list<Tag0, Tag1, Tag2, Tag3>>,
    "Failed testing get_inbox_tags");

static_assert(
    cpp17::is_same_v<Parallel::get_initialization_actions_list<
                         ComponentInit::phase_dependent_action_list>,
                     tmpl::list<InitAction0, InitAction1, InitAction2>>,
    "Failed testing get_intialization_actions_list");

static_assert(
    cpp17::is_same_v<Parallel::get_initialization_actions_list<
                         ComponentExecute::phase_dependent_action_list>,
                     tmpl::list<>>,
    "Failed testing get_intialization_actions_list");

static_assert(
    cpp17::is_same_v<Parallel::get_initialization_actions_list<
                         ComponentInitAndExecute::phase_dependent_action_list>,
                     tmpl::list<InitAction3>>,
    "Failed testing get_intialization_actions_list");

static_assert(cpp17::is_same_v<ComponentInit::initialization_tags,
                               tmpl::list<InitTag0, InitTag1, InitTag2>>,
              "Failed testing get_initialization_tags");

static_assert(
    cpp17::is_same_v<ComponentExecute::initialization_tags, tmpl::list<>>,
    "Failed testing get_initialization_tags");

static_assert(cpp17::is_same_v<ComponentInitAndExecute::initialization_tags,
                               tmpl::list<InitTag0, InitTag1, InitTag3>>,
              "Failed testing get_initialization_tags");

static_assert(cpp17::is_same_v<
                  ComponentInitWithAllocate::initialization_tags,
                  tmpl::list<InitTag4, InitTag5, InitTag0, InitTag1, InitTag2>>,
              "Failed testing get_initialization_tags");

static_assert(
    cpp17::is_same_v<ComponentExecuteWithAllocate::initialization_tags,
                     tmpl::list<InitTag6, InitTag7>>,
    "Failed testing get_initialization_tags");

static_assert(
    cpp17::is_same_v<ComponentInitAndExecuteWithAllocate::initialization_tags,
                     tmpl::list<InitTag3, InitTag4, InitTag0, InitTag1>>,
    "Failed testing get_initialization_tags");

namespace OptionTags {
struct Yards {};
struct Dim {};
struct Greeting {};
struct Name {};
}  // namespace OptionTags

namespace Initialization {
namespace Tags {
struct Yards {
  using option_tags = tmpl::list<OptionTags::Yards>;
};
struct Feet {
  using option_tags = tmpl::list<OptionTags::Yards>;
};
struct Sides {
  using option_tags = tmpl::list<OptionTags::Yards, OptionTags::Dim>;
};
struct FullGreeting {
  using option_tags = tmpl::list<OptionTags::Greeting, OptionTags::Name>;
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

static_assert(cpp17::is_same_v<Parallel::get_option_tags<initialization_tags_0>,
                               tmpl::list<>>,
              "Failed testing get_option_tags");

static_assert(cpp17::is_same_v<Parallel::get_option_tags<initialization_tags_1>,
                               tmpl::list<OptionTags::Yards, OptionTags::Dim>>,
              "Failed testing get_option_tags");

static_assert(
    cpp17::is_same_v<
        Parallel::get_option_tags<initialization_tags_2>,
        tmpl::list<OptionTags::Yards, OptionTags::Greeting, OptionTags::Name>>,
    "Failed testing get_option_tags");

}  // namespace
