// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
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
  using type = db::item_type<Tag>;
};

template <typename Tag, typename Arg1, typename Arg2>
struct PrefixWithArgs : db::PrefixTag, db::SimpleTag {
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
static_assert(cpp17::is_same_v<
                  db::add_tag_prefix<Prefix, PrefixWithArgs<Var, int, double>>,
                  Prefix<PrefixWithArgs<Var, int, double>>>,
              "Failed testing add_tag_prefix");
static_assert(
    cpp17::is_same_v<
        db::add_tag_prefix<
            Prefix, PrefixWithArgs<Tags::Variables<tmpl::list<
                                       PrefixWithArgs<Var, int, double>>>,
                                   int, double>>,
        Prefix<PrefixWithArgs<Tags::Variables<tmpl::list<
                                  Prefix<PrefixWithArgs<Var, int, double>>>>,
                              int, double>>>,
    "Failed testing add_tag_prefix");
static_assert(cpp17::is_same_v<
                  db::add_tag_prefix<PrefixWithArgs, Prefix<Var>, int, double>,
                  PrefixWithArgs<Prefix<Var>, int, double>>,
              "Failed testing add_tag_prefix");
static_assert(
    cpp17::is_same_v<
        db::add_tag_prefix<PrefixWithArgs,
                           Prefix<Tags::Variables<tmpl::list<Prefix<Var>>>>,
                           int, double>,
        PrefixWithArgs<Prefix<Tags::Variables<tmpl::list<
                           PrefixWithArgs<Prefix<Var>, int, double>>>>,
                       int, double>>,
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
    cpp17::is_same_v<db::remove_tag_prefix<PrefixWithArgs<
                         Prefix<Tags::Variables<tmpl::list<
                             PrefixWithArgs<Prefix<Var>, int, double>>>>,
                         int, double>>,
                     Prefix<Tags::Variables<tmpl::list<Prefix<Var>>>>>,
    "Failed testing remove_tag_prefix");

static_assert(
    cpp17::is_same_v<db::remove_all_prefixes<PrefixWithArgs<
                         PrefixWithArgs<Var, int, double>, char, bool>>,
                     Var>,
    "Failed testing remove_all_prefixes");
static_assert(
    cpp17::is_same_v<db::remove_all_prefixes<
                         Prefix<Tags::Variables<tmpl::list<Prefix<Var>>>>>,
                     Tags::Variables<tmpl::list<Var>>>,
    "Failed testing remove_all_prefixes");
static_assert(
    cpp17::is_same_v<db::remove_all_prefixes<PrefixWithArgs<
                         Prefix<Tags::Variables<tmpl::list<
                             PrefixWithArgs<Prefix<Var>, char, bool>>>>,
                         char, bool>>,
                     Tags::Variables<tmpl::list<Var>>>,
    "Failed testing remove_all_prefixes");

static_assert(
    cpp17::is_same_v<
        db::variables_tag_with_tags_list<
            PrefixWithArgs<Prefix<Tags::Variables<tmpl::list<
                               PrefixWithArgs<Prefix<Var>, char, bool>>>>,
                           char, bool>,
            tmpl::list<Var, Prefix<Var>>>,
        PrefixWithArgs<Prefix<Tags::Variables<tmpl::list<Var, Prefix<Var>>>>,
                       char, bool>>,
    "Failed testing variables_tag_with_tags_list");

/// [variables_tag_with_tags_list]
static_assert(
    cpp17::is_same_v<db::variables_tag_with_tags_list<
                         Prefix<Tags::Variables<tmpl::list<Prefix<Var>>>>,
                         tmpl::list<Prefix<Var2>>>,
                     Prefix<Tags::Variables<tmpl::list<Prefix<Var2>>>>>,
    "Failed testing variables_tag_with_tags_list");
/// [variables_tag_with_tags_list]

static_assert(
    cpp17::is_same_v<db::get_variables_tags_list<
                         Prefix<Tags::Variables<tmpl::list<Prefix<Var>>>>>,
                     tmpl::list<Prefix<Var>>>,
    "Failed testing get_variables_tags_list");

static_assert(
    cpp17::is_same_v<
        db::get_variables_tags_list<
            Prefix<Tags::Variables<tmpl::list<Prefix<Var>, Prefix<Var2>>>>>,
        tmpl::list<Prefix<Var>, Prefix<Var2>>>,
    "Failed testing get_variables_tags_list");

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
