// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace TestAddToBox_detail {
struct NoEquivalence {
  int value = 0;
};

struct TagInt : db::SimpleTag {
  static std::string name() noexcept { return "Int"; }
  using type = int;
};

struct TagNoEquivalence : db::SimpleTag {
  static std::string name() noexcept { return "NoEquivalence"; }
  using type = NoEquivalence;
};

template <typename Tag>
struct TagMultiplyByTwo : db::ComputeTag {
  static std::string name() noexcept { return "MultiplyByTwo"; }
  static int function(const int& t) noexcept { return t * 2; }
  static int function(const NoEquivalence& t) noexcept { return t.value * 2; }
  using argument_tags = tmpl::list<Tag>;
};

struct FakeAction {};

void test() {
  auto box0 =
      Initialization::merge_into_databox<FakeAction, db::AddSimpleTags<TagInt>>(
          db::create<db::AddSimpleTags<>>(), 2);
  CHECK(db::get<TagInt>(box0) == 2);
  auto box1 =
      Initialization::merge_into_databox<FakeAction, db::AddSimpleTags<TagInt>>(
          std::move(box0), 2);
  CHECK(db::get<TagInt>(box1) == 2);
  auto box2 =
      Initialization::merge_into_databox<FakeAction,
                                         db::AddSimpleTags<TagNoEquivalence>>(
          std::move(box1), NoEquivalence{5});
  CHECK(db::get<TagNoEquivalence>(box2).value == 5);

  // Make sure that for tags that don't have an equivalence operator we don't
  // overwrite the value.
  auto box3 = Initialization::merge_into_databox<
      FakeAction, db::AddSimpleTags<TagNoEquivalence>, db::AddComputeTags<>,
      Initialization::MergePolicy::IgnoreIncomparable>(std::move(box2),
                                                       NoEquivalence{7});
  CHECK(db::get<TagNoEquivalence>(box3).value == 5);

  // Check that we can overwrite tags that have an equivalence operator
  auto box4 = Initialization::merge_into_databox<
      FakeAction, db::AddSimpleTags<TagInt>, db::AddComputeTags<>,
      Initialization::MergePolicy::Overwrite>(std::move(box3), 4);
  CHECK(db::get<TagInt>(box4) == 4);

  auto box5 = Initialization::merge_into_databox<
      FakeAction, db::AddSimpleTags<TagNoEquivalence>, db::AddComputeTags<>,
      Initialization::MergePolicy::Overwrite>(std::move(box4),
                                              NoEquivalence{7});
  CHECK(db::get<TagNoEquivalence>(box5).value == 7);

  // Check that adding nothing works
  auto box6 =
      Initialization::merge_into_databox<FakeAction, db::AddSimpleTags<>>(
          std::move(box5));
  CHECK(db::get<TagInt>(box6) == 4);
  CHECK(db::get<TagNoEquivalence>(box6).value == 7);

  // Now test that compute items are not re-added, and that they are properly
  // reset if a simple tag is mutated.
  auto box7 = Initialization::merge_into_databox<
      FakeAction, db::AddSimpleTags<>,
      db::AddComputeTags<TagMultiplyByTwo<TagInt>>>(std::move(box6));
  CHECK(db::get<TagMultiplyByTwo<TagInt>>(box7) == 8);

  auto box8 = Initialization::merge_into_databox<
      FakeAction, db::AddSimpleTags<>,
      db::AddComputeTags<TagMultiplyByTwo<TagInt>,
                         TagMultiplyByTwo<TagNoEquivalence>>>(std::move(box7));
  CHECK(db::get<TagMultiplyByTwo<TagNoEquivalence>>(box8) == 14);

  // Now swap out the value of TagInt
  auto box9 = Initialization::merge_into_databox<
      FakeAction, db::AddSimpleTags<TagInt>,
      db::AddComputeTags<TagMultiplyByTwo<TagInt>,
                         TagMultiplyByTwo<TagNoEquivalence>>,
      Initialization::MergePolicy::Overwrite>(std::move(box8), 10);
  CHECK(db::get<TagMultiplyByTwo<TagInt>>(box9) == 20);
}

void test_failure_value() {
  auto box0 =
      Initialization::merge_into_databox<FakeAction, db::AddSimpleTags<TagInt>>(
          db::create<db::AddSimpleTags<>>(), 2);
  CHECK(db::get<TagInt>(box0) == 2);
  // This merge should give an error since they're different values
  auto box1 =
      Initialization::merge_into_databox<FakeAction, db::AddSimpleTags<TagInt>>(
          std::move(box0), 3);
}
}  // namespace TestAddToBox_detail

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Initialization.AddToDataBox",
                  "[Unit][ParallelAlgorithms]") {
  TestAddToBox_detail::test();
}

// [[OutputRegex, While adding the simple tag Int that is already in the DataBox
// we found that the value being set by the action
// TestAddToBox_detail::FakeAction is not the same as what is already in the
// DataBox. The value in the DataBox is: 2 while the value being added is 3]]
SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Initialization.AddToDataBox.Error",
                  "[Unit][ParallelAlgorithms]") {
  ERROR_TEST();
  TestAddToBox_detail::test_failure_value();
}
