// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim, typename T>
struct TestMetavariables {
  static constexpr size_t dim = Dim;
  using stored_type = T;
};

namespace Tags {
struct Dim : db::SimpleTag {
  using type = size_t;
};

struct DimCompute : Dim, db::ComputeTag {
  using base = Dim;
  template <typename Metavars>
  static void function(gsl::not_null<size_t*> result,
                       const Metavars& /*metavars*/) {
    *result = Metavars::dim;
  }
  using argument_tags = tmpl::list<Parallel::Tags::Metavariables>;
};

struct IsString : db::SimpleTag {
  using type = bool;
};
}  // namespace Tags

struct Mutator : tt::ConformsTo<db::protocols::Mutator> {
  using return_tags = tmpl::list<Tags::IsString>;
  using argument_tags = tmpl::list<Parallel::Tags::Metavariables>;

  template <typename Metavars>
  static void apply(const gsl::not_null<bool*> is_string,
                    const Metavars& /*meta*/) {
    *is_string = std::is_same_v<typename Metavars::stored_type, std::string>;
  }
};

static_assert(db::tag_is_retrievable_v<
              Parallel::Tags::Metavariables,
              db::DataBox<tmpl::list<Parallel::Tags::MetavariablesImpl<
                  TestMetavariables<3, std::string>>>>>);

static_assert(not db::tag_is_retrievable_v<Parallel::Tags::Metavariables,
                                           db::DataBox<tmpl::list<Tags::Dim>>>);

static_assert(std::is_same_v<
              const TestMetavariables<9, char>&,
              db::const_item_type<Parallel::Tags::Metavariables,
                                  tmpl::list<Parallel::Tags::MetavariablesImpl<
                                      TestMetavariables<9, char>>>>>);

void test() {
  auto box = db::create<db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<
                                              TestMetavariables<2, double>>,
                                          Tags::IsString>,
                        db::AddComputeTags<Tags::DimCompute>>();
  // Test get
  static_assert(std::is_same_v<
                TestMetavariables<2, double>,
                typename std::decay_t<
                    decltype(db::get<Parallel::Tags::Metavariables>(box))>>);

  // Test compute item
  CHECK(db::get<Tags::Dim>(box) == 2);

  db::mutate<Tags::IsString>(
      [](const gsl::not_null<bool*> is_string) { *is_string = true; },
      make_not_null(&box));
  CHECK(db::get<Tags::IsString>(box));

  // Test mutator
  db::mutate_apply(Mutator{}, make_not_null(&box));
  CHECK_FALSE(db::get<Tags::IsString>(box));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.MetavarialbesTag",
                  "[Unit][DataStructures]") {
  TestHelpers::db::test_simple_tag<
      Parallel::Tags::MetavariablesImpl<TestMetavariables<7, int>>>(
      "Metavariables");
  test();
}
