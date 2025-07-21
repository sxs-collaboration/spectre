// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/TagTraits.hpp"
#include "Helpers/DataStructures/DataBox/TestTags.hpp"

static_assert(not db::is_compute_tag_v<TestHelpers::db::Tags::Bad>);
static_assert(not db::is_compute_tag_v<TestHelpers::db::Tags::Simple>);
static_assert(db::is_compute_tag_v<TestHelpers::db::Tags::SimpleCompute>);
static_assert(not db::is_compute_tag_v<TestHelpers::db::Tags::SimpleReference>);
static_assert(not db::is_compute_tag_v<
              TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

static_assert(not db::is_reference_tag_v<TestHelpers::db::Tags::Bad>);
static_assert(not db::is_reference_tag_v<TestHelpers::db::Tags::Simple>);
static_assert(not db::is_reference_tag_v<TestHelpers::db::Tags::SimpleCompute>);
static_assert(db::is_reference_tag_v<TestHelpers::db::Tags::SimpleReference>);
static_assert(not db::is_reference_tag_v<
              TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

static_assert(not db::is_immutable_item_tag_v<TestHelpers::db::Tags::Bad>);
static_assert(not db::is_immutable_item_tag_v<TestHelpers::db::Tags::Simple>);
static_assert(
    db::is_immutable_item_tag_v<TestHelpers::db::Tags::SimpleCompute>);
static_assert(
    db::is_immutable_item_tag_v<TestHelpers::db::Tags::SimpleReference>);
static_assert(not db::is_immutable_item_tag_v<
              TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

static_assert(not db::is_mutable_item_tag_v<TestHelpers::db::Tags::Bad>);
static_assert(db::is_mutable_item_tag_v<TestHelpers::db::Tags::Simple>);
static_assert(
    not db::is_mutable_item_tag_v<TestHelpers::db::Tags::SimpleCompute>);
static_assert(
    not db::is_mutable_item_tag_v<TestHelpers::db::Tags::SimpleReference>);
static_assert(db::is_mutable_item_tag_v<
              TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

static_assert(not db::is_simple_tag_v<TestHelpers::db::Tags::Bad>);
static_assert(db::is_simple_tag_v<TestHelpers::db::Tags::Simple>);
static_assert(not db::is_simple_tag_v<TestHelpers::db::Tags::SimpleCompute>);
static_assert(not db::is_simple_tag_v<TestHelpers::db::Tags::SimpleReference>);
static_assert(db::is_simple_tag_v<
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
