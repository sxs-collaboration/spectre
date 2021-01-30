// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/TagTraits.hpp"
#include "Helpers/DataStructures/DataBox/TestTags.hpp"

static_assert(not db::is_compute_tag_v<TestHelpers::db::Tags::Bad>,
              "Failed testing is_compute_tag");
static_assert(not db::is_compute_tag_v<TestHelpers::db::Tags::Simple>,
              "Failed testing is_compute_tag");
static_assert(not db::is_compute_tag_v<TestHelpers::db::Tags::Base>,
              "Failed testing is_compute_tag");
static_assert(not db::is_compute_tag_v<TestHelpers::db::Tags::SimpleWithBase>,
              "Failed testing is_compute_tag");
static_assert(db::is_compute_tag_v<TestHelpers::db::Tags::SimpleCompute>,
              "Failed testing is_compute_tag");
static_assert(
    db::is_compute_tag_v<TestHelpers::db::Tags::SimpleWithBaseCompute>,
    "Failed testing is_compute_tag");
static_assert(not db::is_compute_tag_v<TestHelpers::db::Tags::SimpleReference>,
              "Failed testing is_compute_tag");
static_assert(
    not db::is_compute_tag_v<TestHelpers::db::Tags::SimpleWithBaseReference>,
    "Failed testing is_compute_tag");
static_assert(not db::is_compute_tag_v<
                  TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>,
              "Failed testing is_compute_tag");

static_assert(not db::is_reference_tag_v<TestHelpers::db::Tags::Bad>,
              "Failed testing is_reference_tag");
static_assert(not db::is_reference_tag_v<TestHelpers::db::Tags::Simple>,
              "Failed testing is_reference_tag");
static_assert(not db::is_reference_tag_v<TestHelpers::db::Tags::Base>,
              "Failed testing is_reference_tag");
static_assert(not db::is_reference_tag_v<TestHelpers::db::Tags::SimpleWithBase>,
              "Failed testing is_reference_tag");
static_assert(not db::is_reference_tag_v<TestHelpers::db::Tags::SimpleCompute>,
              "Failed testing is_reference_tag");
static_assert(
    not db::is_reference_tag_v<TestHelpers::db::Tags::SimpleWithBaseCompute>,
    "Failed testing is_reference_tag");
static_assert(db::is_reference_tag_v<TestHelpers::db::Tags::SimpleReference>,
              "Failed testing is_reference_tag");
static_assert(
    db::is_reference_tag_v<TestHelpers::db::Tags::SimpleWithBaseReference>,
    "Failed testing is_reference_tag");
static_assert(not db::is_reference_tag_v<
                  TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>,
              "Failed testing is_reference_tag");

static_assert(not db::is_immutable_item_tag_v<TestHelpers::db::Tags::Bad>,
              "Failed testing is_immutable_item_tag");
static_assert(not db::is_immutable_item_tag_v<TestHelpers::db::Tags::Simple>,
              "Failed testing is_immutable_item_tag");
static_assert(not db::is_immutable_item_tag_v<TestHelpers::db::Tags::Base>,
              "Failed testing is_immutable_item_tag");
static_assert(
    not db::is_immutable_item_tag_v<TestHelpers::db::Tags::SimpleWithBase>,
    "Failed testing is_immutable_item_tag");
static_assert(db::is_immutable_item_tag_v<TestHelpers::db::Tags::SimpleCompute>,
              "Failed testing is_immutable_item_tag");
static_assert(
    db::is_immutable_item_tag_v<TestHelpers::db::Tags::SimpleWithBaseCompute>,
    "Failed testing is_immutable_item_tag");
static_assert(
    db::is_immutable_item_tag_v<TestHelpers::db::Tags::SimpleReference>,
    "Failed testing is_immutable_item_tag");
static_assert(
    db::is_immutable_item_tag_v<TestHelpers::db::Tags::SimpleWithBaseReference>,
    "Failed testing is_immutable_item_tag");
static_assert(not db::is_immutable_item_tag_v<
                  TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>,
              "Failed testing is_immutable_item_tag");

static_assert(not db::is_simple_tag_v<TestHelpers::db::Tags::Bad>,
              "Failed testing is_simple_tag");
static_assert(db::is_simple_tag_v<TestHelpers::db::Tags::Simple>,
              "Failed testing is_simple_tag");
static_assert(not db::is_simple_tag_v<TestHelpers::db::Tags::Base>,
              "Failed testing is_simple_tag");
static_assert(db::is_simple_tag_v<TestHelpers::db::Tags::SimpleWithBase>,
              "Failed testing is_simple_tag");
static_assert(not db::is_simple_tag_v<TestHelpers::db::Tags::SimpleCompute>,
              "Failed testing is_simple_tag");
static_assert(
    not db::is_simple_tag_v<TestHelpers::db::Tags::SimpleWithBaseCompute>,
    "Failed testing is_simple_tag");
static_assert(not db::is_simple_tag_v<TestHelpers::db::Tags::SimpleReference>,
              "Failed testing is_simple_tag");
static_assert(
    not db::is_simple_tag_v<TestHelpers::db::Tags::SimpleWithBaseReference>,
    "Failed testing is_simple_tag");
static_assert(db::is_simple_tag_v<
                  TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>,
              "Failed testing is_simple_tag");

static_assert(not db::is_non_base_tag_v<TestHelpers::db::Tags::Bad>,
              "Failed testing is_non_base_tag");
static_assert(db::is_non_base_tag_v<TestHelpers::db::Tags::Simple>,
              "Failed testing is_non_base_tag");
static_assert(not db::is_non_base_tag_v<TestHelpers::db::Tags::Base>,
              "Failed testing is_non_base_tag");
static_assert(db::is_non_base_tag_v<TestHelpers::db::Tags::SimpleWithBase>,
              "Failed testing is_non_base_tag");
static_assert(db::is_non_base_tag_v<TestHelpers::db::Tags::SimpleCompute>,
              "Failed testing is_non_base_tag");
static_assert(
    db::is_non_base_tag_v<TestHelpers::db::Tags::SimpleWithBaseCompute>,
    "Failed testing is_non_base_tag");
static_assert(db::is_non_base_tag_v<TestHelpers::db::Tags::SimpleReference>,
              "Failed testing is_non_base_tag");
static_assert(
    db::is_non_base_tag_v<TestHelpers::db::Tags::SimpleWithBaseReference>,
    "Failed testing is_non_base_tag");
static_assert(db::is_non_base_tag_v<
                  TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>,
              "Failed testing is_non_base_tag");

static_assert(not db::is_tag_v<TestHelpers::db::Tags::Bad>,
              "Failed testing is_tag");
static_assert(db::is_tag_v<TestHelpers::db::Tags::Simple>,
              "Failed testing is_tag");
static_assert(db::is_tag_v<TestHelpers::db::Tags::Base>,
              "Failed testing is_tag");
static_assert(db::is_tag_v<TestHelpers::db::Tags::SimpleWithBase>,
              "Failed testing is_tag");
static_assert(db::is_tag_v<TestHelpers::db::Tags::SimpleCompute>,
              "Failed testing is_tag");
static_assert(db::is_tag_v<TestHelpers::db::Tags::SimpleWithBaseCompute>,
              "Failed testing is_tag");
static_assert(db::is_tag_v<TestHelpers::db::Tags::SimpleReference>,
              "Failed testing is_tag");
static_assert(db::is_tag_v<TestHelpers::db::Tags::SimpleWithBaseReference>,
              "Failed testing is_tag");
static_assert(
    db::is_tag_v<TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>,
    "Failed testing is_tag");

static_assert(not db::is_base_tag_v<TestHelpers::db::Tags::Bad>,
              "Failed testing is_base_tag");
static_assert(not db::is_base_tag_v<TestHelpers::db::Tags::Simple>,
              "Failed testing is_base_tag");
static_assert(db::is_base_tag_v<TestHelpers::db::Tags::Base>,
              "Failed testing is_base_tag");
static_assert(not db::is_base_tag_v<TestHelpers::db::Tags::SimpleWithBase>,
              "Failed testing is_base_tag");
static_assert(not db::is_base_tag_v<TestHelpers::db::Tags::SimpleCompute>,
              "Failed testing is_base_tag");
static_assert(
    not db::is_base_tag_v<TestHelpers::db::Tags::SimpleWithBaseCompute>,
    "Failed testing is_base_tag");
static_assert(not db::is_base_tag_v<TestHelpers::db::Tags::SimpleReference>,
              "Failed testing is_base_tag");
static_assert(
    not db::is_base_tag_v<TestHelpers::db::Tags::SimpleWithBaseReference>,
    "Failed testing is_base_tag");
static_assert(not db::is_base_tag_v<
                  TestHelpers::db::Tags::Label<TestHelpers::db::Tags::Simple>>,
              "Failed testing is_base_tag");
