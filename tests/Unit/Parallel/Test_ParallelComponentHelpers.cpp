// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/ParallelComponentHelpers.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
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

struct InitAction0 {};
struct InitAction1 {};
struct InitAction2 {};
struct InitAction3 {};

enum class Phase { Initialization, Execute, Exit };

struct ComponentInit {
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Phase, Phase::Initialization,
      tmpl::list<InitAction0, InitAction1, InitAction2>>>;
};

struct ComponentExecute {
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<Phase, Phase::Execute,
                                        tmpl::list<Action0, Action1>>>;
};

struct ComponentInitAndExecute {
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Phase, Phase::Initialization,
                             tmpl::list<InitAction3>>,
      Parallel::PhaseActions<Phase, Phase::Execute,
                             tmpl::list<InitAction2, Action0, Action2>>>;
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

}  // namespace
