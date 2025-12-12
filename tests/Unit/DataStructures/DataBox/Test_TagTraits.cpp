// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/TagTraits.hpp"
#include "Helpers/DataStructures/DataBox/TestTags.hpp"

static_assert(not db::is_compute_tag_v<TestHelpers::db::Tags::Bad>);
static_assert(not db::is_compute_tag_v<TestHelpers::db::Tags::Simple>);
static_assert(db::is_compute_tag_v<TestHelpers::db::Tags::SimpleCompute>);
static_assert(not db::is_compute_tag_v<TestHelpers::db::Tags::SimpleReference>);
static_assert(not db::is_compute_tag_v<
              TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

static_assert(not db::compute_tag<TestHelpers::db::Tags::Bad>);
static_assert(not db::compute_tag<TestHelpers::db::Tags::Simple>);
static_assert(db::compute_tag<TestHelpers::db::Tags::SimpleCompute>);
static_assert(not db::compute_tag<TestHelpers::db::Tags::SimpleReference>);
static_assert(not db::compute_tag<
              TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

static_assert(not db::is_reference_tag_v<TestHelpers::db::Tags::Bad>);
static_assert(not db::is_reference_tag_v<TestHelpers::db::Tags::Simple>);
static_assert(not db::is_reference_tag_v<TestHelpers::db::Tags::SimpleCompute>);
static_assert(db::is_reference_tag_v<TestHelpers::db::Tags::SimpleReference>);
static_assert(not db::is_reference_tag_v<
              TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

static_assert(not db::reference_tag<TestHelpers::db::Tags::Bad>);
static_assert(not db::reference_tag<TestHelpers::db::Tags::Simple>);
static_assert(not db::reference_tag<TestHelpers::db::Tags::SimpleCompute>);
static_assert(db::reference_tag<TestHelpers::db::Tags::SimpleReference>);
static_assert(not db::reference_tag<
              TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

static_assert(not db::is_immutable_item_tag_v<TestHelpers::db::Tags::Bad>);
static_assert(not db::is_immutable_item_tag_v<TestHelpers::db::Tags::Simple>);
static_assert(
    db::is_immutable_item_tag_v<TestHelpers::db::Tags::SimpleCompute>);
static_assert(
    db::is_immutable_item_tag_v<TestHelpers::db::Tags::SimpleReference>);
static_assert(not db::is_immutable_item_tag_v<
              TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

static_assert(not db::immutable_item_tag<TestHelpers::db::Tags::Bad>);
static_assert(not db::immutable_item_tag<TestHelpers::db::Tags::Simple>);
static_assert(db::immutable_item_tag<TestHelpers::db::Tags::SimpleCompute>);
static_assert(db::immutable_item_tag<TestHelpers::db::Tags::SimpleReference>);
static_assert(not db::immutable_item_tag<
              TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

static_assert(not db::is_mutable_item_tag_v<TestHelpers::db::Tags::Bad>);
static_assert(db::is_mutable_item_tag_v<TestHelpers::db::Tags::Simple>);
static_assert(
    not db::is_mutable_item_tag_v<TestHelpers::db::Tags::SimpleCompute>);
static_assert(
    not db::is_mutable_item_tag_v<TestHelpers::db::Tags::SimpleReference>);
static_assert(db::is_mutable_item_tag_v<
              TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

static_assert(not db::mutable_item_tag<TestHelpers::db::Tags::Bad>);
static_assert(db::mutable_item_tag<TestHelpers::db::Tags::Simple>);
static_assert(not db::mutable_item_tag<TestHelpers::db::Tags::SimpleCompute>);
static_assert(not db::mutable_item_tag<TestHelpers::db::Tags::SimpleReference>);
static_assert(db::mutable_item_tag<
              TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

static_assert(not db::is_simple_tag_v<TestHelpers::db::Tags::Bad>);
static_assert(db::is_simple_tag_v<TestHelpers::db::Tags::Simple>);
static_assert(not db::is_simple_tag_v<TestHelpers::db::Tags::SimpleCompute>);
static_assert(not db::is_simple_tag_v<TestHelpers::db::Tags::SimpleReference>);
static_assert(db::is_simple_tag_v<
              TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

static_assert(not db::simple_tag<TestHelpers::db::Tags::Bad>);
static_assert(db::simple_tag<TestHelpers::db::Tags::Simple>);
static_assert(not db::simple_tag<TestHelpers::db::Tags::SimpleCompute>);
static_assert(not db::simple_tag<TestHelpers::db::Tags::SimpleReference>);
static_assert(db::simple_tag<
              TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

static_assert(not db::is_creation_tag_v<TestHelpers::db::Tags::Bad>);
static_assert(db::is_creation_tag_v<TestHelpers::db::Tags::Simple>);
static_assert(db::is_creation_tag_v<TestHelpers::db::Tags::SimpleCompute>);
static_assert(db::is_creation_tag_v<TestHelpers::db::Tags::SimpleReference>);
static_assert(db::is_creation_tag_v<
              TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

static_assert(not db::is_tag_v<TestHelpers::db::Tags::Bad>);
static_assert(db::is_tag_v<TestHelpers::db::Tags::Simple>);
static_assert(db::is_tag_v<TestHelpers::db::Tags::SimpleCompute>);
static_assert(db::is_tag_v<TestHelpers::db::Tags::SimpleReference>);
static_assert(
    db::is_tag_v<TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

static_assert(not db::tag<TestHelpers::db::Tags::Bad>);
static_assert(db::tag<TestHelpers::db::Tags::Simple>);
static_assert(db::tag<TestHelpers::db::Tags::SimpleCompute>);
static_assert(db::tag<TestHelpers::db::Tags::SimpleReference>);
static_assert(
    db::tag<TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

namespace {
enum class TagType1 { Simple, Compute, Reference, Other };

template <db::tag Tag>
struct ConceptTest1 {
  static constexpr TagType1 value = TagType1::Other;
};

template <db::simple_tag Tag>
struct ConceptTest1<Tag> {
  static constexpr TagType1 value = TagType1::Simple;
};

template <db::compute_tag Tag>
struct ConceptTest1<Tag> {
  static constexpr TagType1 value = TagType1::Compute;
};

template <db::reference_tag Tag>
struct ConceptTest1<Tag> {
  static constexpr TagType1 value = TagType1::Reference;
};

static_assert(ConceptTest1<TestHelpers::db::Tags::Simple>::value ==
              TagType1::Simple);
static_assert(ConceptTest1<TestHelpers::db::Tags::SimpleCompute>::value ==
              TagType1::Compute);
static_assert(ConceptTest1<TestHelpers::db::Tags::SimpleReference>::value ==
              TagType1::Reference);
static_assert(ConceptTest1<Parallel::Tags::Metavariables>::value ==
              TagType1::Other);

enum class TagType2 { Compute, Immutable, Tag };

template <db::tag Tag>
struct ConceptTest2 {
  static constexpr TagType2 value = TagType2::Tag;
};

template <db::immutable_item_tag Tag>
struct ConceptTest2<Tag> {
  static constexpr TagType2 value = TagType2::Immutable;
};

template <db::compute_tag Tag>
struct ConceptTest2<Tag> {
  static constexpr TagType2 value = TagType2::Compute;
};

static_assert(ConceptTest2<TestHelpers::db::Tags::Simple>::value ==
              TagType2::Tag);
static_assert(ConceptTest2<TestHelpers::db::Tags::SimpleCompute>::value ==
              TagType2::Compute);
static_assert(ConceptTest2<TestHelpers::db::Tags::SimpleReference>::value ==
              TagType2::Immutable);
static_assert(ConceptTest2<Parallel::Tags::Metavariables>::value ==
              TagType2::Tag);
}  // namespace
