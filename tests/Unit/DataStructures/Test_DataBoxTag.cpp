// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <type_traits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

class DataVector;

namespace {
struct Var : db::DataBoxTag {
  using type = Scalar<DataVector>;
};

template <typename Tag>
struct Prefix : db::DataBoxPrefix {
  using tag = Tag;
  using type = db::item_type<Tag>;
};

template <typename Tag, typename Arg1, typename Arg2>
struct PrefixWithArgs : db::DataBoxPrefix {
  using tag = Tag;
  using type = db::item_type<Tag>;
};
}  // namespace

static_assert(cpp17::is_same_v<db::add_tag_prefix<Prefix, Var>, Prefix<Var>>,
              "Failed testing add_tag_prefix");
static_assert(cpp17::is_same_v<
                  db::add_tag_prefix<Prefix, Tags::Variables<tmpl::list<Var>>>,
                  Prefix<Tags::Variables<tmpl::list<Prefix<Var>>>>>,
              "Failed testing add_tag_prefix");
static_assert(
    cpp17::is_same_v<db::add_tag_prefix<PrefixWithArgs, Var, int, double>,
                     PrefixWithArgs<Var, int, double>>,
    "Failed testing add_tag_prefix");
static_assert(
    cpp17::is_same_v<
        db::add_tag_prefix<PrefixWithArgs, Tags::Variables<tmpl::list<Var>>,
                           int, double>,
        PrefixWithArgs<
            Tags::Variables<tmpl::list<PrefixWithArgs<Var, int, double>>>, int,
            double>>,
    "Failed testing add_tag_prefix");

static_assert(cpp17::is_same_v<db::remove_tag_prefix<Prefix<Var>>, Var>,
              "Failed testing remove_tag_prefix");
static_assert(
    cpp17::is_same_v<
        db::remove_tag_prefix<Prefix<Tags::Variables<tmpl::list<Prefix<Var>>>>>,
        Tags::Variables<tmpl::list<Var>>>,
    "Failed testing remove_tag_prefix");
static_assert(cpp17::is_same_v<
                  db::remove_tag_prefix<PrefixWithArgs<Var, int, double>>, Var>,
              "Failed testing remove_tag_prefix");
static_assert(
    cpp17::is_same_v<
        db::remove_tag_prefix<PrefixWithArgs<
            Tags::Variables<tmpl::list<PrefixWithArgs<Var, int, double>>>, int,
            double>>,
        Tags::Variables<tmpl::list<Var>>>,
    "Failed testing remove_tag_prefix");

static_assert(
    cpp17::is_same_v<db::remove_all_prefixes<PrefixWithArgs<
                         PrefixWithArgs<Var, int, double>, char, bool>>,
                     Var>,
    "Failed testing remove_tag_prefix");
