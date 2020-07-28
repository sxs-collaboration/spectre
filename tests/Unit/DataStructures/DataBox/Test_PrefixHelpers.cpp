// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Helpers/DataStructures/TestTags.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

class DataVector;

namespace {
using Var = TestHelpers::Tags::Scalar<>;

using Var2 = TestHelpers::Tags::Scalar2<>;

template <typename Tag>
using Prefix = TestHelpers::Tags::Prefix1<Tag>;

template <typename Tag>
using Prefix2 = TestHelpers::Tags::Prefix2<Tag>;

template <typename Arg0, typename Arg1, typename Arg2>
struct SomeType;

template <typename Tag, typename Arg1, typename Arg2>
struct PrefixWithArgs : db::PrefixTag, db::SimpleTag {
  using tag = Tag;
  using type = SomeType<typename Tag::type, Arg1, Arg2>;
};

using vars_list = tmpl::list<Var, Var2>;
using prefix_vars_list = tmpl::list<Prefix<Var>, Prefix<Var2>>;
using double_prefix_vars_list =
    tmpl::list<Prefix<Prefix<Var>>, Prefix<Prefix<Var2>>>;
using both_prefix_vars_list =
    tmpl::list<Prefix2<Prefix<Var>>, Prefix2<Prefix<Var2>>>;

template <typename Arg1, typename Arg2>
using args_vars_list = tmpl::list<PrefixWithArgs<Var, Arg1, Arg2>,
                                  PrefixWithArgs<Var2, Arg1, Arg2>>;

template <typename Arg1, typename Arg2>
using args_prefix_vars_list =
    tmpl::list<PrefixWithArgs<Prefix<Var>, Arg1, Arg2>,
               PrefixWithArgs<Prefix<Var2>, Arg1, Arg2>>;

template <typename Arg1, typename Arg2>
using prefix_args_vars_list =
    tmpl::list<Prefix<PrefixWithArgs<Var, Arg1, Arg2>>,
               Prefix<PrefixWithArgs<Var2, Arg1, Arg2>>>;
}  // namespace

// Test db::wrap_tags_in
static_assert(
    std::is_same_v<db::wrap_tags_in<Prefix, vars_list>, prefix_vars_list>);
static_assert(
    std::is_same_v<db::wrap_tags_in<PrefixWithArgs, vars_list, int, double>,
                   args_vars_list<int, double>>);
static_assert(std::is_same_v<db::wrap_tags_in<Prefix, prefix_vars_list>,
                             double_prefix_vars_list>);
static_assert(std::is_same_v<db::wrap_tags_in<Prefix2, prefix_vars_list>,
                             both_prefix_vars_list>);
static_assert(std::is_same_v<
              db::wrap_tags_in<PrefixWithArgs, prefix_vars_list, int, double>,
              args_prefix_vars_list<int, double>>);
static_assert(
    std::is_same_v<db::wrap_tags_in<Prefix, args_vars_list<int, double>>,
                   prefix_args_vars_list<int, double>>);

// Test db::add_tag_prefix on tag
static_assert(std::is_same_v<db::add_tag_prefix<Prefix, Var>, Prefix<Var>>);
static_assert(
    std::is_same_v<db::add_tag_prefix<PrefixWithArgs, Var, int, double>,
                   PrefixWithArgs<Var, int, double>>);
static_assert(std::is_same_v<db::add_tag_prefix<Prefix, Prefix<Var>>,
                             Prefix<Prefix<Var>>>);
static_assert(std::is_same_v<db::add_tag_prefix<Prefix2, Prefix<Var>>,
                             Prefix2<Prefix<Var>>>);
static_assert(
    std::is_same_v<db::add_tag_prefix<Prefix, PrefixWithArgs<Var, int, double>>,
                   Prefix<PrefixWithArgs<Var, int, double>>>);
static_assert(
    std::is_same_v<db::add_tag_prefix<PrefixWithArgs, Prefix<Var>, int, double>,
                   PrefixWithArgs<Prefix<Var>, int, double>>);

namespace {
using vars_tag = Tags::Variables<vars_list>;
using prefix_vars_tag = Tags::Variables<prefix_vars_list>;
using double_prefix_vars_tag = Tags::Variables<double_prefix_vars_list>;
using both_prefix_vars_tag = Tags::Variables<both_prefix_vars_list>;
template <typename Arg1, typename Arg2>
using args_vars_tag = Tags::Variables<args_vars_list<Arg1, Arg2>>;
template <typename Arg1, typename Arg2>
using args_prefix_vars_tag = Tags::Variables<args_prefix_vars_list<Arg1, Arg2>>;
template <typename Arg1, typename Arg2>
using prefix_args_vars_tag = Tags::Variables<prefix_args_vars_list<Arg1, Arg2>>;
}  // namespace

// Test db::add_tag_prefix on Variables tag
static_assert(
    std::is_same_v<db::add_tag_prefix<Prefix, vars_tag>, prefix_vars_tag>);
static_assert(
    std::is_same_v<db::add_tag_prefix<PrefixWithArgs, vars_tag, int, double>,
                   args_vars_tag<int, double>>);
static_assert(std::is_same_v<db::add_tag_prefix<Prefix, prefix_vars_tag>,
                             double_prefix_vars_tag>);
static_assert(std::is_same_v<db::add_tag_prefix<Prefix2, prefix_vars_tag>,
                             both_prefix_vars_tag>);
static_assert(
    std::is_same_v<db::add_tag_prefix<Prefix, args_vars_tag<int, double>>,
                   prefix_args_vars_tag<int, double>>);
static_assert(std::is_same_v<
              db::add_tag_prefix<PrefixWithArgs, prefix_vars_tag, int, double>,
              args_prefix_vars_tag<int, double>>);

// Test db::remove_tag_prefix on tag
static_assert(std::is_same_v<db::remove_tag_prefix<Prefix<Var>>, Var>);
static_assert(std::is_same_v<
              db::remove_tag_prefix<PrefixWithArgs<Var, int, double>>, Var>);
static_assert(
    std::is_same_v<db::remove_tag_prefix<Prefix<Prefix<Var>>>, Prefix<Var>>);
static_assert(
    std::is_same_v<db::remove_tag_prefix<Prefix2<Prefix<Var>>>, Prefix<Var>>);
static_assert(std::is_same_v<
              db::remove_tag_prefix<Prefix<PrefixWithArgs<Var, int, double>>>,
              PrefixWithArgs<Var, int, double>>);
static_assert(std::is_same_v<
              db::remove_tag_prefix<PrefixWithArgs<Prefix<Var>, int, double>>,
              Prefix<Var>>);

// Test db::remove_tag_prefix on Variables tag
static_assert(std::is_same_v<db::remove_tag_prefix<prefix_vars_tag>, vars_tag>);
static_assert(std::is_same_v<db::remove_tag_prefix<args_vars_tag<int, double>>,
                             vars_tag>);
static_assert(std::is_same_v<db::remove_tag_prefix<double_prefix_vars_tag>,
                             prefix_vars_tag>);
static_assert(std::is_same_v<db::remove_tag_prefix<both_prefix_vars_tag>,
                             prefix_vars_tag>);
static_assert(
    std::is_same_v<db::remove_tag_prefix<prefix_args_vars_tag<int, double>>,
                   args_vars_tag<int, double>>);
static_assert(
    std::is_same_v<db::remove_tag_prefix<args_prefix_vars_tag<int, double>>,
                   prefix_vars_tag>);

// Test db::remove_all_prefixes on tag
static_assert(std::is_same_v<db::remove_all_prefixes<Prefix<Var>>, Var>);
static_assert(std::is_same_v<
              db::remove_all_prefixes<PrefixWithArgs<Var, int, double>>, Var>);
static_assert(
    std::is_same_v<db::remove_all_prefixes<Prefix<Prefix<Var>>>, Var>);
static_assert(
    std::is_same_v<db::remove_all_prefixes<Prefix2<Prefix<Var>>>, Var>);
static_assert(std::is_same_v<
              db::remove_all_prefixes<Prefix<PrefixWithArgs<Var, int, double>>>,
              Var>);
static_assert(std::is_same_v<
              db::remove_all_prefixes<PrefixWithArgs<Prefix<Var>, int, double>>,
              Var>);

// Test db::remove_all_prefixes on Variables tag
static_assert(
    std::is_same_v<db::remove_all_prefixes<prefix_vars_tag>, vars_tag>);
static_assert(std::is_same_v<
              db::remove_all_prefixes<args_vars_tag<int, double>>, vars_tag>);
static_assert(
    std::is_same_v<db::remove_all_prefixes<double_prefix_vars_tag>, vars_tag>);
static_assert(
    std::is_same_v<db::remove_all_prefixes<both_prefix_vars_tag>, vars_tag>);
static_assert(
    std::is_same_v<db::remove_all_prefixes<prefix_args_vars_tag<int, double>>,
                   vars_tag>);
static_assert(
    std::is_same_v<db::remove_all_prefixes<args_prefix_vars_tag<int, double>>,
                   vars_tag>);
