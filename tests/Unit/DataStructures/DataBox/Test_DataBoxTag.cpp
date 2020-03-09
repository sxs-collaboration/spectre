// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

class DataVector;

namespace {
struct Var : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct Var2 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <typename Tag>
struct Prefix : db::PrefixTag, db::SimpleTag {
  using tag = Tag;
  using type = db::const_item_type<Tag>;
};

template <typename Tag, typename Arg1, typename Arg2>
struct PrefixWithArgs : db::PrefixTag, db::SimpleTag {
  using tag = Tag;
  using type = db::const_item_type<Tag>;
};
}  // namespace

static_assert(cpp17::is_same_v<db::split_tag<Var>, tmpl::list<Var>>,
              "Failed testing split_tag");
static_assert(
    cpp17::is_same_v<db::split_tag<Prefix<Var>>, tmpl::list<Prefix<Var>>>,
    "Failed testing split_tag");
static_assert(cpp17::is_same_v<
                  db::split_tag<Tags::Variables<tmpl::list<Var, Prefix<Var>>>>,
                  tmpl::list<Var, Prefix<Var>>>,
              "Failed testing split_tag");
static_assert(
    cpp17::is_same_v<
        db::split_tag<Prefix<Tags::Variables<tmpl::list<Var, Prefix<Var>>>>>,
        tmpl::list<Var, Prefix<Var>>>,
    "Failed testing split_tag");

/// [prefix_tag_wraps_specified_tag]
static_assert(db::prefix_tag_wraps_specified_tag<Prefix<Var>, Var>::value,
              "failed testing prefix_tag_wraps_specified_tag");

static_assert(not db::prefix_tag_wraps_specified_tag<Prefix<Var2>, Var>::value,
              "failed testing prefix_tag_wraps_specified_tag");

static_assert(db::prefix_tag_wraps_specified_tag<PrefixWithArgs<Var, int, int>,
                                                 Var>::value,
              "failed testing prefix_tag_wraps_specified_tag");

static_assert(
    cpp17::is_same_v<tmpl::filter<tmpl::list<Prefix<Var>, Prefix<Var2>,
                                             PrefixWithArgs<Var, int, int>>,
                                  db::prefix_tag_wraps_specified_tag<
                                      tmpl::_1, tmpl::pin<Var>>>,
                     tmpl::list<Prefix<Var>, PrefixWithArgs<Var, int, int>>>,
    "failed testing prefix_tag_wraps_specified_tag");
/// [prefix_tag_wraps_specified_tag]
