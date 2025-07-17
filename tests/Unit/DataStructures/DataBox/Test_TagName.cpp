// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Helpers/DataStructures/DataBox/TestTags.hpp"

namespace {
struct NamedSimple : db::SimpleTag {
  static std::string name() { return "NameOfSimple"; }
};

struct SimpleNamedCompute : TestHelpers::db::Tags::Simple, db::ComputeTag {
  static std::string name() { return "NameOfSimpleCompute"; }
};

struct NamedSimpleNamedCompute : NamedSimple, db::ComputeTag {
  static std::string name() { return "NameOfNamedSimpleCompute"; }
};

struct NamedSimpleCompute : NamedSimple, db::ComputeTag {
  using base = NamedSimple;
};

struct SimpleNamedReference : TestHelpers::db::Tags::Simple, db::ReferenceTag {
  static std::string name() { return "NameOfSimpleReference"; }
};

struct NamedSimpleNamedReference : NamedSimple, db::ReferenceTag {
  static std::string name() { return "NameOfNamedSimpleReference"; }
};

struct NamedSimpleReference : NamedSimple, db::ReferenceTag {
  using base = NamedSimple;
};

template <typename Tag>
struct NamedLabel : db::PrefixTag, db::SimpleTag {
  using tag = Tag;
  static std::string name() {
    return "NameOfLabel(" + db::tag_name<Tag>() + ")";
  }
};

template <typename Tag>
struct LabelNamedCompute : TestHelpers::db::Tags::Label<Tag>, db::ComputeTag {
  using tag = Tag;
  static std::string name() {
    return "NameOfLabelCompute(" + db::tag_name<Tag>() + ")";
  }
};

template <typename Tag>
struct NamedLabelCompute : NamedLabel<Tag>, db::ComputeTag {
  using base = NamedLabel<Tag>;
  using tag = Tag;
};

template <typename Tag>
struct NamedLabelNamedCompute : NamedLabel<Tag>, db::ComputeTag {
  using tag = Tag;
  static std::string name() {
    return "NameOfNamedLabelCompute(" + db::tag_name<Tag>() + ")";
  }
};

}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.TagName",
                  "[Unit][DataStructures]") {
  CHECK(db::tag_name<TestHelpers::db::Tags::Simple>() == "Simple");
  CHECK(db::tag_name<TestHelpers::db::Tags::SimpleCompute>() == "Simple");
  CHECK(db::tag_name<TestHelpers::db::Tags::SimpleReference>() == "Simple");
  CHECK(db::tag_name<NamedSimple>() == "NameOfSimple");
  CHECK(db::tag_name<SimpleNamedCompute>() == "NameOfSimpleCompute");
  CHECK(db::tag_name<NamedSimpleNamedCompute>() == "NameOfNamedSimpleCompute");
  CHECK(db::tag_name<NamedSimpleCompute>() == "NameOfSimple");
  CHECK(db::tag_name<SimpleNamedReference>() == "NameOfSimpleReference");
  CHECK(db::tag_name<NamedSimpleNamedReference>() ==
        "NameOfNamedSimpleReference");
  CHECK(db::tag_name<NamedSimpleReference>() == "NameOfSimple");

  CHECK(db::tag_name<
            TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>() ==
        "Label(Simple)");
  CHECK(db::tag_name<TestHelpers::db::Tags::Label<
            TestHelpers::db::Tags::SimpleCompute>>() == "Label(Simple)");
  CHECK(db::tag_name<TestHelpers::db::Tags::Label<NamedSimple>>() ==
        "Label(NameOfSimple)");
  CHECK(db::tag_name<TestHelpers::db::Tags::Label<SimpleNamedCompute>>() ==
        "Label(NameOfSimpleCompute)");
  CHECK(db::tag_name<TestHelpers::db::Tags::Label<NamedSimpleNamedCompute>>() ==
        "Label(NameOfNamedSimpleCompute)");
  CHECK(db::tag_name<TestHelpers::db::Tags::Label<NamedSimpleCompute>>() ==
        "Label(NameOfSimple)");

  CHECK(db::tag_name<NamedLabel<TestHelpers::db::Tags::Simple>>() ==
        "NameOfLabel(Simple)");
  CHECK(db::tag_name<LabelNamedCompute<TestHelpers::db::Tags::Simple>>() ==
        "NameOfLabelCompute(Simple)");
  CHECK(db::tag_name<NamedLabelCompute<TestHelpers::db::Tags::Simple>>() ==
        "NameOfLabel(Simple)");
  CHECK(db::tag_name<NamedLabelNamedCompute<TestHelpers::db::Tags::Simple>>() ==
        "NameOfNamedLabelCompute(Simple)");

  CHECK(db::tag_name<NamedLabel<NamedSimple>>() == "NameOfLabel(NameOfSimple)");
  CHECK(db::tag_name<LabelNamedCompute<NamedSimple>>() ==
        "NameOfLabelCompute(NameOfSimple)");
  CHECK(db::tag_name<NamedLabelCompute<NamedSimple>>() ==
        "NameOfLabel(NameOfSimple)");
  CHECK(db::tag_name<NamedLabelNamedCompute<NamedSimple>>() ==
        "NameOfNamedLabelCompute(NameOfSimple)");
}
