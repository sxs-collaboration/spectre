// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/TagTraits.hpp"
#include "Helpers/DataStructures/DataBox/TestTags.hpp"

static_assert(not db::is_compute_tag_v<TestHelpers::db::Tags::Bad>);
static_assert(not db::is_compute_tag_v<TestHelpers::db::Tags::Simple>);
static_assert(not db::is_compute_tag_v<TestHelpers::db::Tags::Base>);
static_assert(not db::is_compute_tag_v<TestHelpers::db::Tags::SimpleWithBase>);
static_assert(db::is_compute_tag_v<TestHelpers::db::Tags::SimpleCompute>);
static_assert(
    db::is_compute_tag_v<TestHelpers::db::Tags::SimpleWithBaseCompute>);
static_assert(not db::is_compute_tag_v<TestHelpers::db::Tags::SimpleReference>);
static_assert(
    not db::is_compute_tag_v<TestHelpers::db::Tags::SimpleWithBaseReference>);
static_assert(not db::is_compute_tag_v<
              TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

static_assert(not db::is_reference_tag_v<TestHelpers::db::Tags::Bad>);
static_assert(not db::is_reference_tag_v<TestHelpers::db::Tags::Simple>);
static_assert(not db::is_reference_tag_v<TestHelpers::db::Tags::Base>);
static_assert(
    not db::is_reference_tag_v<TestHelpers::db::Tags::SimpleWithBase>);
static_assert(not db::is_reference_tag_v<TestHelpers::db::Tags::SimpleCompute>);
static_assert(
    not db::is_reference_tag_v<TestHelpers::db::Tags::SimpleWithBaseCompute>);
static_assert(db::is_reference_tag_v<TestHelpers::db::Tags::SimpleReference>);
static_assert(
    db::is_reference_tag_v<TestHelpers::db::Tags::SimpleWithBaseReference>);
static_assert(not db::is_reference_tag_v<
              TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

static_assert(not db::is_immutable_item_tag_v<TestHelpers::db::Tags::Bad>);
static_assert(not db::is_immutable_item_tag_v<TestHelpers::db::Tags::Simple>);
static_assert(not db::is_immutable_item_tag_v<TestHelpers::db::Tags::Base>);
static_assert(
    not db::is_immutable_item_tag_v<TestHelpers::db::Tags::SimpleWithBase>);
static_assert(
    db::is_immutable_item_tag_v<TestHelpers::db::Tags::SimpleCompute>);
static_assert(
    db::is_immutable_item_tag_v<TestHelpers::db::Tags::SimpleWithBaseCompute>);
static_assert(
    db::is_immutable_item_tag_v<TestHelpers::db::Tags::SimpleReference>);
static_assert(db::is_immutable_item_tag_v<
              TestHelpers::db::Tags::SimpleWithBaseReference>);
static_assert(not db::is_immutable_item_tag_v<
              TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

static_assert(not db::is_simple_tag_v<TestHelpers::db::Tags::Bad>);
static_assert(db::is_simple_tag_v<TestHelpers::db::Tags::Simple>);
static_assert(not db::is_simple_tag_v<TestHelpers::db::Tags::Base>);
static_assert(db::is_simple_tag_v<TestHelpers::db::Tags::SimpleWithBase>);
static_assert(not db::is_simple_tag_v<TestHelpers::db::Tags::SimpleCompute>);
static_assert(
    not db::is_simple_tag_v<TestHelpers::db::Tags::SimpleWithBaseCompute>);
static_assert(not db::is_simple_tag_v<TestHelpers::db::Tags::SimpleReference>);
static_assert(
    not db::is_simple_tag_v<TestHelpers::db::Tags::SimpleWithBaseReference>);
static_assert(db::is_simple_tag_v<
              TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

static_assert(not db::is_non_base_tag_v<TestHelpers::db::Tags::Bad>);
static_assert(db::is_non_base_tag_v<TestHelpers::db::Tags::Simple>);
static_assert(not db::is_non_base_tag_v<TestHelpers::db::Tags::Base>);
static_assert(db::is_non_base_tag_v<TestHelpers::db::Tags::SimpleWithBase>);
static_assert(db::is_non_base_tag_v<TestHelpers::db::Tags::SimpleCompute>);
static_assert(
    db::is_non_base_tag_v<TestHelpers::db::Tags::SimpleWithBaseCompute>);
static_assert(db::is_non_base_tag_v<TestHelpers::db::Tags::SimpleReference>);
static_assert(
    db::is_non_base_tag_v<TestHelpers::db::Tags::SimpleWithBaseReference>);
static_assert(db::is_non_base_tag_v<
              TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

static_assert(not db::is_tag_v<TestHelpers::db::Tags::Bad>);
static_assert(db::is_tag_v<TestHelpers::db::Tags::Simple>);
static_assert(db::is_tag_v<TestHelpers::db::Tags::Base>);
static_assert(db::is_tag_v<TestHelpers::db::Tags::SimpleWithBase>);
static_assert(db::is_tag_v<TestHelpers::db::Tags::SimpleCompute>);
static_assert(db::is_tag_v<TestHelpers::db::Tags::SimpleWithBaseCompute>);
static_assert(db::is_tag_v<TestHelpers::db::Tags::SimpleReference>);
static_assert(db::is_tag_v<TestHelpers::db::Tags::SimpleWithBaseReference>);
static_assert(
    db::is_tag_v<TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);

static_assert(not db::is_base_tag_v<TestHelpers::db::Tags::Bad>);
static_assert(not db::is_base_tag_v<TestHelpers::db::Tags::Simple>);
static_assert(db::is_base_tag_v<TestHelpers::db::Tags::Base>);
static_assert(not db::is_base_tag_v<TestHelpers::db::Tags::SimpleWithBase>);
static_assert(not db::is_base_tag_v<TestHelpers::db::Tags::SimpleCompute>);
static_assert(
    not db::is_base_tag_v<TestHelpers::db::Tags::SimpleWithBaseCompute>);
static_assert(not db::is_base_tag_v<TestHelpers::db::Tags::SimpleReference>);
static_assert(
    not db::is_base_tag_v<TestHelpers::db::Tags::SimpleWithBaseReference>);
static_assert(not db::is_base_tag_v<
              TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>);
