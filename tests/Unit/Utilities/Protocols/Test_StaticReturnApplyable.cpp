// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Protocols/StaticReturnApplyable.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct T1 {};
struct T2 {};
struct T3 {};
struct T4 {};

struct Tag1 : db::SimpleTag {
  using type = T1;
};

struct Tag2 : db::SimpleTag {
  using type = T2;
};

template <typename T>
struct TemplatedTag : db::SimpleTag {
  using type = T;
};

// [StaticReturnApplyable]
struct Applyable : tt::ConformsTo<protocols::StaticReturnApplyable> {
  using return_tags = tmpl::list<Tag1, TemplatedTag<T3>>;
  using argument_tags = tmpl::list<TemplatedTag<T4>, Tag2>;
  static void apply(const gsl::not_null<T1*> /*return1*/,
                    const gsl::not_null<T3*> /*return2*/,
                    const T4& /*argument1*/, const T2& /*argument2*/) {}
};
// [StaticReturnApplyable]

static_assert(
    tt::assert_conforms_to_v<Applyable, protocols::StaticReturnApplyable>);

// new struct where apply is a template function
struct ApplyableTemplatedApply
    : tt::ConformsTo<protocols::StaticReturnApplyable> {
  using return_tags = tmpl::list<Tag1, TemplatedTag<T3>>;
  using argument_tags = tmpl::list<TemplatedTag<T4>, Tag2>;

  template <typename T>
  static void apply(const gsl::not_null<T1*> /*return1*/,
                    const gsl::not_null<T3*> /*return2*/,
                    const T4& /*argument1*/, const T& /*argument2*/) {}
};

static_assert(tt::assert_conforms_to_v<ApplyableTemplatedApply,
                                       protocols::StaticReturnApplyable>);

}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.Protocols.StaticReturnApplyable",
                  "[Unit][Utilities]") {
  // Make sure the protocol is consistent with db::mutate_apply.  Not
  // all valid arguments to db::mutate_apply satisfy this protocol,
  // but the reverse should be true.
  auto box = db::create<
      db::AddSimpleTags<Tag1, Tag2, TemplatedTag<T3>, TemplatedTag<T4>>>(
      T1{}, T2{}, T3{}, T4{});
  db::mutate_apply<Applyable>(make_not_null(&box));
  db::mutate_apply<ApplyableTemplatedApply>(make_not_null(&box));
  CHECK(true);
}
