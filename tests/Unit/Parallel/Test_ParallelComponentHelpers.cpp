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

static_assert(cpp17::is_same_v<Parallel::get_inbox_tags_from_action<Action2>,
                               tmpl::list<Tag1, Tag2, Tag3>>,
              "Failed testing get_inbox_tags_from_action");

static_assert(
    cpp17::is_same_v<
        Parallel::get_inbox_tags<tmpl::list<Action0, Action1, Action2>>,
        tmpl::list<Tag0, Tag1, Tag2, Tag3>>,
    "Failed testing get_inbox_tags");
}  // namespace
