// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagTraits.hpp"
#include "Parallel/InitializationTag.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct NormalTag : db::SimpleTag {
  using type = int;
};

struct InitTag1 : db::SimpleTag {
  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<>;
};

struct InitTag2 : db::SimpleTag {
  static constexpr bool pass_metavariables = true;
  template <typename Metavariables>
  using option_tags =
      tmpl::conditional_t<Metavariables::something, tmpl::list<>, tmpl::list<>>;
};

struct BadInitTag1 : db::SimpleTag {
  static constexpr bool pass_metavariables = false;
};

struct BadInitTag2 : db::SimpleTag {
  using option_tags = tmpl::list<>;
};

struct BadInitTag3 : db::SimpleTag {
  static constexpr bool pass_metavariables = true;
  using option_tags = tmpl::list<>;
};

struct BadInitTag4 : db::SimpleTag {
  static constexpr bool pass_metavariables = false;
  template <typename Metavariables>
  using option_tags =
      tmpl::conditional_t<Metavariables::something, tmpl::list<>, tmpl::list<>>;
};

static_assert(not Parallel::initialization_tag<NormalTag>);
static_assert(Parallel::initialization_tag<InitTag1>);
static_assert(Parallel::initialization_tag<InitTag2>);

static_assert(not Parallel::templated_initialization_tag<NormalTag>);
static_assert(not Parallel::templated_initialization_tag<InitTag1>);
static_assert(Parallel::templated_initialization_tag<InitTag2>);

static_assert(not Parallel::untemplated_initialization_tag<NormalTag>);
static_assert(Parallel::untemplated_initialization_tag<InitTag1>);
static_assert(not Parallel::untemplated_initialization_tag<InitTag2>);

static_assert(not Parallel::initialization_tag<BadInitTag1>);
static_assert(not Parallel::initialization_tag<BadInitTag2>);
static_assert(not Parallel::initialization_tag<BadInitTag3>);
static_assert(not Parallel::initialization_tag<BadInitTag4>);
}  // namespace
