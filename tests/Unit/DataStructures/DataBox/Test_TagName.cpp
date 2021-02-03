// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Helpers/DataStructures/DataBox/TestTags.hpp"

namespace {
struct NamedSimple : db::SimpleTag {
  static std::string name() noexcept { return "NameOfSimple"; }
};

struct NamedBase : db::BaseTag {
  static std::string name() noexcept { return "NameOfBase"; }
};

struct NamedSimpleWithBase : TestHelpers::db::Tags::Base, db::SimpleTag {
  static std::string name() noexcept { return "NameOfSimpleWithBase"; }
};

struct SimpleWithNamedBase : NamedBase, db::SimpleTag {};

struct NamedSimpleWithNamedBase : NamedBase, db::SimpleTag {
  static std::string name() noexcept { return "NameOfSimpleWithNamedBase"; }
};

struct SimpleNamedCompute : TestHelpers::db::Tags::Simple, db::ComputeTag {
  static std::string name() noexcept { return "NameOfSimpleCompute"; }
};

struct NamedSimpleNamedCompute : NamedSimple, db::ComputeTag {
  static std::string name() noexcept { return "NameOfNamedSimpleCompute"; }
};

struct NamedSimpleCompute : NamedSimple, db::ComputeTag {
  using base = NamedSimple;
};

struct SimpleWithBaseNamedCompute : TestHelpers::db::Tags::SimpleWithBase,
                                    db::ComputeTag {
  static std::string name() noexcept { return "NameOfSimpleWithBaseCompute"; }
};

struct NamedSimpleWithBaseNamedCompute : NamedSimpleWithBase, db::ComputeTag {
  static std::string name() noexcept {
    return "NameOfNamedSimpleWithBaseCompute";
  }
};

struct NamedSimpleWithBaseCompute : NamedSimpleWithBase, db::ComputeTag {
  using base = NamedSimpleWithBase;
};

struct SimpleWithNamedBaseNamedCompute : SimpleWithNamedBase, db::ComputeTag {
  static std::string name() noexcept {
    return "NameOfSimpleWithNamedBaseCompute";
  }
};

struct SimpleWithNamedBaseCompute : SimpleWithNamedBase, db::ComputeTag {
  using base = SimpleWithNamedBase;
};

struct NamedSimpleWithNamedBaseNamedCompute : NamedSimpleWithNamedBase,
                                              db::ComputeTag {
  static std::string name() noexcept {
    return "NameOfNamedSimpleWithNamedBaseCompute";
  }
};

struct NamedSimpleWithNamedBaseCompute : NamedSimpleWithNamedBase,
                                         db::ComputeTag {
  using base = NamedSimpleWithNamedBase;
};

struct SimpleNamedReference : TestHelpers::db::Tags::Simple, db::ReferenceTag {
  static std::string name() noexcept { return "NameOfSimpleReference"; }
};

struct NamedSimpleNamedReference : NamedSimple, db::ReferenceTag {
  static std::string name() noexcept { return "NameOfNamedSimpleReference"; }
};

struct NamedSimpleReference : NamedSimple, db::ReferenceTag {
  using base = NamedSimple;
};

struct SimpleWithBaseNamedReference : TestHelpers::db::Tags::SimpleWithBase,
                                      db::ReferenceTag {
  static std::string name() noexcept { return "NameOfSimpleWithBaseReference"; }
};

struct NamedSimpleWithBaseNamedReference : NamedSimpleWithBase,
                                           db::ReferenceTag {
  static std::string name() noexcept {
    return "NameOfNamedSimpleWithBaseReference";
  }
};

struct NamedSimpleWithBaseReference : NamedSimpleWithBase, db::ReferenceTag {
  using base = NamedSimpleWithBase;
};

struct SimpleWithNamedBaseNamedReference : SimpleWithNamedBase,
                                           db::ReferenceTag {
  static std::string name() noexcept {
    return "NameOfSimpleWithNamedBaseReference";
  }
};

struct SimpleWithNamedBaseReference : SimpleWithNamedBase, db::ReferenceTag {
  using base = SimpleWithNamedBase;
};

struct NamedSimpleWithNamedBaseNamedReference : NamedSimpleWithNamedBase,
                                                db::ReferenceTag {
  static std::string name() noexcept {
    return "NameOfNamedSimpleWithNamedBaseReference";
  }
};

struct NamedSimpleWithNamedBaseReference : NamedSimpleWithNamedBase,
                                           db::ReferenceTag {
  using base = NamedSimpleWithNamedBase;
};

template <typename Tag>
struct NamedLabel : db::PrefixTag, db::SimpleTag {
  using tag = Tag;
  static std::string name() noexcept {
    return "NameOfLabel(" + db::tag_name<Tag>() + ")";
  }
};

template <typename Tag>
struct LabelNamedCompute : TestHelpers::db::Tags::Label<Tag>, db::ComputeTag {
  using tag = Tag;
  static std::string name() noexcept {
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
  static std::string name() noexcept {
    return "NameOfNamedLabelCompute(" + db::tag_name<Tag>() + ")";
  }
};

}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.TagName",
                  "[Unit][DataStructures]") {
  CHECK(db::tag_name<TestHelpers::db::Tags::Base>() == "Base");
  CHECK(db::tag_name<TestHelpers::db::Tags::Simple>() == "Simple");
  CHECK(db::tag_name<TestHelpers::db::Tags::SimpleWithBase>() ==
        "SimpleWithBase");
  CHECK(db::tag_name<TestHelpers::db::Tags::SimpleCompute>() == "Simple");
  CHECK(db::tag_name<TestHelpers::db::Tags::SimpleWithBaseCompute>() ==
        "SimpleWithBase");
  CHECK(db::tag_name<TestHelpers::db::Tags::SimpleReference>() == "Simple");
  CHECK(db::tag_name<TestHelpers::db::Tags::SimpleWithBaseReference>() ==
        "SimpleWithBase");
  CHECK(db::tag_name<NamedSimple>() == "NameOfSimple");
  CHECK(db::tag_name<NamedBase>() == "NameOfBase");
  CHECK(db::tag_name<NamedSimpleWithBase>() == "NameOfSimpleWithBase");
  CHECK(db::tag_name<SimpleWithNamedBase>() == "NameOfBase");
  CHECK(db::tag_name<NamedSimpleWithNamedBase>() ==
        "NameOfSimpleWithNamedBase");
  CHECK(db::tag_name<SimpleNamedCompute>() == "NameOfSimpleCompute");
  CHECK(db::tag_name<NamedSimpleNamedCompute>() == "NameOfNamedSimpleCompute");
  CHECK(db::tag_name<NamedSimpleCompute>() == "NameOfSimple");
  CHECK(db::tag_name<SimpleWithBaseNamedCompute>() ==
        "NameOfSimpleWithBaseCompute");
  CHECK(db::tag_name<NamedSimpleWithBaseNamedCompute>() ==
        "NameOfNamedSimpleWithBaseCompute");
  CHECK(db::tag_name<NamedSimpleWithBaseCompute>() == "NameOfSimpleWithBase");
  CHECK(db::tag_name<SimpleWithNamedBaseNamedCompute>() ==
        "NameOfSimpleWithNamedBaseCompute");
  CHECK(db::tag_name<SimpleWithNamedBaseCompute>() == "NameOfBase");
  CHECK(db::tag_name<NamedSimpleWithNamedBaseNamedCompute>() ==
        "NameOfNamedSimpleWithNamedBaseCompute");
  CHECK(db::tag_name<NamedSimpleWithNamedBaseCompute>() ==
        "NameOfSimpleWithNamedBase");
  CHECK(db::tag_name<SimpleNamedReference>() == "NameOfSimpleReference");
  CHECK(db::tag_name<NamedSimpleNamedReference>() ==
        "NameOfNamedSimpleReference");
  CHECK(db::tag_name<NamedSimpleReference>() == "NameOfSimple");
  CHECK(db::tag_name<SimpleWithBaseNamedReference>() ==
        "NameOfSimpleWithBaseReference");
  CHECK(db::tag_name<NamedSimpleWithBaseNamedReference>() ==
        "NameOfNamedSimpleWithBaseReference");
  CHECK(db::tag_name<NamedSimpleWithBaseReference>() == "NameOfSimpleWithBase");
  CHECK(db::tag_name<SimpleWithNamedBaseNamedReference>() ==
        "NameOfSimpleWithNamedBaseReference");
  CHECK(db::tag_name<SimpleWithNamedBaseReference>() == "NameOfBase");
  CHECK(db::tag_name<NamedSimpleWithNamedBaseNamedReference>() ==
        "NameOfNamedSimpleWithNamedBaseReference");
  CHECK(db::tag_name<NamedSimpleWithNamedBaseReference>() ==
        "NameOfSimpleWithNamedBase");

  CHECK(db::tag_name<
            TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Base>>() ==
        "Label(Base)");
  CHECK(db::tag_name<
            TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>() ==
        "Label(Simple)");
  CHECK(db::tag_name<TestHelpers::db::Tags::Label<
            TestHelpers::db::Tags::SimpleWithBase>>() ==
        "Label(SimpleWithBase)");
  CHECK(db::tag_name<TestHelpers::db::Tags::Label<
            TestHelpers::db::Tags::SimpleCompute>>() == "Label(Simple)");
  CHECK(db::tag_name<TestHelpers::db::Tags::Label<
            TestHelpers::db::Tags::SimpleWithBaseCompute>>() ==
        "Label(SimpleWithBase)");
  CHECK(db::tag_name<TestHelpers::db::Tags::Label<NamedSimple>>() ==
        "Label(NameOfSimple)");
  CHECK(db::tag_name<TestHelpers::db::Tags::Label<NamedBase>>() ==
        "Label(NameOfBase)");
  CHECK(db::tag_name<TestHelpers::db::Tags::Label<NamedSimpleWithBase>>() ==
        "Label(NameOfSimpleWithBase)");
  CHECK(db::tag_name<TestHelpers::db::Tags::Label<SimpleWithNamedBase>>() ==
        "Label(NameOfBase)");
  CHECK(
      db::tag_name<TestHelpers::db::Tags::Label<NamedSimpleWithNamedBase>>() ==
      "Label(NameOfSimpleWithNamedBase)");
  CHECK(db::tag_name<TestHelpers::db::Tags::Label<SimpleNamedCompute>>() ==
        "Label(NameOfSimpleCompute)");
  CHECK(db::tag_name<TestHelpers::db::Tags::Label<NamedSimpleNamedCompute>>() ==
        "Label(NameOfNamedSimpleCompute)");
  CHECK(db::tag_name<TestHelpers::db::Tags::Label<NamedSimpleCompute>>() ==
        "Label(NameOfSimple)");
  CHECK(db::tag_name<
            TestHelpers::db::Tags::Label<SimpleWithBaseNamedCompute>>() ==
        "Label(NameOfSimpleWithBaseCompute)");
  CHECK(db::tag_name<
            TestHelpers::db::Tags::Label<NamedSimpleWithBaseNamedCompute>>() ==
        "Label(NameOfNamedSimpleWithBaseCompute)");
  CHECK(db::tag_name<
            TestHelpers::db::Tags::Label<NamedSimpleWithBaseCompute>>() ==
        "Label(NameOfSimpleWithBase)");
  CHECK(db::tag_name<
            TestHelpers::db::Tags::Label<SimpleWithNamedBaseNamedCompute>>() ==
        "Label(NameOfSimpleWithNamedBaseCompute)");
  CHECK(db::tag_name<
            TestHelpers::db::Tags::Label<SimpleWithNamedBaseCompute>>() ==
        "Label(NameOfBase)");
  CHECK(db::tag_name<TestHelpers::db::Tags::Label<
            NamedSimpleWithNamedBaseNamedCompute>>() ==
        "Label(NameOfNamedSimpleWithNamedBaseCompute)");
  CHECK(db::tag_name<
            TestHelpers::db::Tags::Label<NamedSimpleWithNamedBaseCompute>>() ==
        "Label(NameOfSimpleWithNamedBase)");

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
