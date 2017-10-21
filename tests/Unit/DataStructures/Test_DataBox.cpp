// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "DataStructures/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBoxHelpers.hpp"
#include "DataStructures/DataBoxTag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Literals.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
double multiply_by_two(const double value) { return 2.0 * value; }
std::string append_word(const std::string& text, const double value) {
  std::stringstream ss;
  ss << value;
  return text + ss.str();
}

auto get_tensor() { return tnsr::A<double, 3, Frame::Grid>{{{7.82, 8, 3, 9}}}; }
}  // namespace

namespace test_databox_tags {
/// [databox_tag_example]
struct Tag0 : db::DataBoxTag {
  using type = double;
};
/// [databox_tag_example]
struct Tag1 : db::DataBoxTag {
  using type = std::vector<double>;
};
struct Tag2 : db::DataBoxTag {
  using type = std::string;
};
struct Tag3 : db::DataBoxTag {
  using type = std::string;
};

/// [databox_compute_item_tag_example]
struct ComputeTag0 : db::ComputeItemTag {
  static constexpr auto function = multiply_by_two;
  using argument_tags = typelist<Tag0>;
};
/// [databox_compute_item_tag_example]
struct ComputeTag1 : db::ComputeItemTag {
  static constexpr auto function = append_word;
  using argument_tags = typelist<Tag2, ComputeTag0>;
};

struct TagTensor : db::ComputeItemTag {
  static constexpr auto function = get_tensor;
  using argument_tags = typelist<>;
};

/// [compute_item_tag_function]
struct ComputeLambda0 : db::ComputeItemTag {
  static constexpr double function(const double a) { return 3.0 * a; }
  using argument_tags = typelist<Tag0>;
};
/// [compute_item_tag_function]

/// [compute_item_tag_no_tags]
struct ComputeLambda1 : db::ComputeItemTag {
  static constexpr double function() { return 7.0; }
  using argument_tags = typelist<>;
};
/// [compute_item_tag_no_tags]

/// [databox_prefix_tag_example]
template <typename Tag>
struct TagPrefix : db::DataBoxPrefix {
  using type = typename Tag::type;
  using tag = Tag;
};
/// [databox_prefix_tag_example]
}  // namespace test_databox_tags

namespace {
template <typename T>
struct X {};
/// [remove_tags_from_keep_tags]
using full_list = tmpl::list<double, char, int, bool, X<int>>;
using keep_list = tmpl::list<double, bool>;
static_assert(
    std::is_same<typelist<char, int, X<int>>,
                 db::remove_tags_from_keep_tags<full_list, keep_list>>::value,
    "Failed testing db::remove_tags_from_keep_tags");
using keep_list2 = tmpl::list<double, bool, X<int>>;
static_assert(
    std::is_same<typelist<char, int>,
                 db::remove_tags_from_keep_tags<full_list, keep_list2>>::value,
    "Failed testing db::remove_tags_from_keep_tags");
/// [remove_tags_from_keep_tags]

using Box_t = db::DataBox<db::get_databox_list<typelist<
    test_databox_tags::Tag0, test_databox_tags::Tag1, test_databox_tags::Tag2,
    test_databox_tags::TagPrefix<test_databox_tags::Tag0>,
    test_databox_tags::ComputeTag0, test_databox_tags::ComputeTag1>>>;

static_assert(
    std::is_same<
        decltype(
            db::create_from<db::RemoveTags<test_databox_tags::Tag1>>(Box_t{})),
        db::DataBox<db::get_databox_list<typelist<
            test_databox_tags::Tag0, test_databox_tags::Tag2,
            test_databox_tags::TagPrefix<test_databox_tags::Tag0>,
            test_databox_tags::ComputeTag0, test_databox_tags::ComputeTag1>>>>::
        value,
    "Failed testing removal of item");

static_assert(
    std::is_same<
        decltype(db::create_from<
                 db::RemoveTags<test_databox_tags::ComputeTag1>>(Box_t{})),
        db::DataBox<db::get_databox_list<typelist<
            test_databox_tags::Tag0, test_databox_tags::Tag1,
            test_databox_tags::TagPrefix<test_databox_tags::Tag0>,
            test_databox_tags::Tag2, test_databox_tags::ComputeTag0>>>>::value,
    "Failed testing removal of compute item");

static_assert(std::is_same<decltype(db::create_from<db::RemoveTags<>>(Box_t{})),
                           Box_t>::value,
              "Failed testing no-op create_from");
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.get_tag_name",
                  "[Unit][DataStructures]") {
  CHECK(db::get_tag_name<test_databox_tags::TagTensor>() == "TagTensor");
  CHECK(db::get_tag_name<
            test_databox_tags::TagPrefix<test_databox_tags::Tag2>>() ==
        "TagPrefixTag2");
}

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox", "[Unit][DataStructures]") {
  /// [create_databox]
  auto original_box =
      db::create<db::AddTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                             test_databox_tags::Tag2>,
                 db::AddComputeItemsTags<test_databox_tags::ComputeTag0,
                                         test_databox_tags::ComputeTag1,
                                         test_databox_tags::ComputeLambda0,
                                         test_databox_tags::ComputeLambda1>>(
          3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
  /// [create_databox]
  static_assert(
      std::is_same<
          decltype(original_box),
          db::DataBox<db::get_databox_list<typelist<
              test_databox_tags::Tag0, test_databox_tags::Tag1,
              test_databox_tags::Tag2, test_databox_tags::ComputeTag0,
              test_databox_tags::ComputeTag1, test_databox_tags::ComputeLambda0,
              test_databox_tags::ComputeLambda1>>>>::value,
      "Failed to create original_box");

  CHECK(db::get<test_databox_tags::Tag0>(original_box) == 3.14);
  // Check retrieving chained compute item result
  CHECK(db::get<test_databox_tags::ComputeTag1>(original_box) ==
        "My Sample String6.28"s);
  CHECK(db::get<test_databox_tags::ComputeLambda0>(original_box) == 3.0 * 3.14);
  CHECK(db::get<test_databox_tags::ComputeLambda1>(original_box) == 7.0);
  // No removal
  {
    auto box = db::create_from<db::RemoveTags<>>(original_box);
    CHECK(db::get<test_databox_tags::Tag2>(box) == "My Sample String"s);
    CHECK(db::get<test_databox_tags::ComputeTag1>(box) ==
          "My Sample String6.28"s);
    CHECK(db::get<test_databox_tags::ComputeLambda0>(box) == 3.0 * 3.14);
    CHECK(db::get<test_databox_tags::ComputeLambda1>(original_box) == 7.0);
  }
  {
    /// [create_from_remove]
    auto box =
        db::create_from<db::RemoveTags<test_databox_tags::Tag1>>(original_box);
    /// [create_from_remove]
    CHECK(db::get<test_databox_tags::Tag2>(box) == "My Sample String"s);
    CHECK(db::get<test_databox_tags::ComputeTag1>(box) ==
          "My Sample String6.28"s);
    CHECK(db::get<test_databox_tags::ComputeLambda0>(box) == 3.0 * 3.14);
    CHECK(db::get<test_databox_tags::ComputeLambda1>(original_box) == 7.0);
  }
  {
    /// [create_from_add_item]
    auto box =
        db::create_from<db::RemoveTags<>, db::AddTags<test_databox_tags::Tag3>>(
            original_box, "Yet another test string"s);
    /// [create_from_add_item]
    CHECK(db::get<test_databox_tags::Tag3>(box) == "Yet another test string"s);
    CHECK(db::get<test_databox_tags::Tag2>(box) == "My Sample String"s);
    // Check retrieving compute item result
    CHECK(db::get<test_databox_tags::ComputeTag1>(box) ==
          "My Sample String6.28"s);
    CHECK(db::get<test_databox_tags::ComputeLambda0>(box) == 3.0 * 3.14);
    CHECK(db::get<test_databox_tags::ComputeLambda1>(original_box) == 7.0);
  }
  {
    /// [create_from_add_compute_item]
    auto simple_box =
        db::create<db::AddTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                               test_databox_tags::Tag2>>(
            3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
    auto box = db::create_from<
        db::RemoveTags<>, db::AddTags<>,
        db::AddComputeItemsTags<test_databox_tags::ComputeTag0>>(simple_box);
    /// [create_from_add_compute_item]
    CHECK(db::get<test_databox_tags::Tag2>(box) == "My Sample String"s);
    // Check retrieving compute item result
    CHECK(db::get<test_databox_tags::ComputeTag0>(box) == 6.28);
  }
  {
    auto simple_box =
        db::create<db::AddTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                               test_databox_tags::Tag2>>(
            3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
    auto box = db::create_from<
        db::RemoveTags<>, db::AddTags<test_databox_tags::Tag3>,
        db::AddComputeItemsTags<test_databox_tags::ComputeTag0>>(
        simple_box, "Yet another test string"s);
    CHECK(db::get<test_databox_tags::Tag3>(box) == "Yet another test string"s);
    CHECK(db::get<test_databox_tags::Tag2>(box) == "My Sample String"s);
    // Check retrieving compute item result
    CHECK(db::get<test_databox_tags::ComputeTag0>(box) == 6.28);
  }
  {
    auto simple_box =
        db::create<db::AddTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                               test_databox_tags::Tag2>>(
            3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
    auto box = db::create_from<
        db::RemoveTags<test_databox_tags::Tag1>,
        db::AddTags<test_databox_tags::Tag3>,
        db::AddComputeItemsTags<test_databox_tags::ComputeTag0>>(
        simple_box, "Yet another test string"s);
    CHECK(db::get<test_databox_tags::Tag3>(box) == "Yet another test string"s);
    CHECK(db::get<test_databox_tags::Tag2>(box) == "My Sample String"s);
    // Check retrieving compute item result
    CHECK(6.28 == db::get<test_databox_tags::ComputeTag0>(box));
  }
}

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.get_item_from_box",
                  "[Unit][DataStructures]") {
  /// [get_item_from_box]
  auto original_box = db::create<
      db::AddTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                  test_databox_tags::Tag2,
                  test_databox_tags::TagPrefix<test_databox_tags::Tag0>>,
      db::AddComputeItemsTags<test_databox_tags::ComputeTag0,
                              test_databox_tags::ComputeTag1>>(
      3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s, 8.7);
  const std::string compute_string =
      db::get_item_from_box<std::string>(original_box, "ComputeTag1");
  /// [get_item_from_box]
  CHECK(compute_string == "My Sample String6.28"s);
  const std::string added_string =
      db::get_item_from_box<std::string>(original_box, "Tag2");
  CHECK(added_string == "My Sample String"s);
  /// [databox_name_prefix]
  CHECK(db::get_item_from_box<double>(original_box, "TagPrefixTag0") == 8.7);
  /// [databox_name_prefix]
}

// [[OutputRegex, Could not find the tag named "time__" in the DataBox]]
SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.get_item_from_box_error_name",
                  "[Unit][DataStructures]") {
  ERROR_TEST();
  auto original_box =
      db::create<db::AddTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                             test_databox_tags::Tag2>,
                 db::AddComputeItemsTags<test_databox_tags::ComputeTag0,
                                         test_databox_tags::ComputeTag1>>(
          3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
  static_cast<void>(db::get_item_from_box<double>(original_box, "time__"));
}

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.apply",
                  "[Unit][DataStructures]") {
  auto original_box =
      db::create<db::AddTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                             test_databox_tags::Tag2>,
                 db::AddComputeItemsTags<test_databox_tags::ComputeTag0,
                                         test_databox_tags::ComputeTag1>>(
          3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
  auto check_result_no_args = [](const std::string& sample_string,
                                 const auto& computed_string) {
    CHECK(sample_string == "My Sample String"s);
    CHECK(computed_string == "My Sample String6.28"s);
  };
  db::apply<typelist<test_databox_tags::Tag2, test_databox_tags::ComputeTag1>>(
      check_result_no_args, original_box);

  /// [apply_example]
  auto check_result_args = [](const std::string& sample_string,
                              const auto& computed_string, const auto& vector) {
    CHECK(sample_string == "My Sample String"s);
    CHECK(computed_string == "My Sample String6.28"s);
    CHECK(vector == (std::vector<double>{8.7, 93.2, 84.7}));
  };
  db::apply<typelist<test_databox_tags::Tag2, test_databox_tags::ComputeTag1>>(
      check_result_args, original_box,
      db::get<test_databox_tags::Tag1>(original_box));
  /// [apply_example]
}

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.apply_with_box",
                  "[Unit][DataStructures]") {
  auto original_box =
      db::create<db::AddTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                             test_databox_tags::Tag2>,
                 db::AddComputeItemsTags<test_databox_tags::ComputeTag0,
                                         test_databox_tags::ComputeTag1>>(
          3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
  auto check_result_no_args = [](const auto& box,
                                 const std::string& sample_string,
                                 const auto& computed_string) {
    CHECK(sample_string == "My Sample String"s);
    CHECK(computed_string == "My Sample String6.28"s);
    CHECK(db::get<test_databox_tags::Tag1>(box) ==
          (std::vector<double>{8.7, 93.2, 84.7}));
  };
  db::apply_with_box<
      typelist<test_databox_tags::Tag2, test_databox_tags::ComputeTag1>>(
      check_result_no_args, original_box);

  /// [apply_with_box_example]
  auto check_result_args = [](const auto& box, const std::string& sample_string,
                              const std::string& computed_string,
                              const std::vector<int>& vector) {
    CHECK(sample_string == "My Sample String"s);
    CHECK(computed_string == "My Sample String6.28"s);
    CHECK(db::get<test_databox_tags::Tag1>(box) ==
          (std::vector<double>{8.7, 93.2, 84.7}));
    CHECK((vector == std::vector<int>{1, 4, 8}));
  };
  db::apply_with_box<
      typelist<test_databox_tags::Tag2, test_databox_tags::ComputeTag1>>(
      check_result_args, original_box, std::vector<int>{1, 4, 8});
  /// [apply_with_box_example]
}

// [[OutputRegex, Could not find the tag named "TagTensor__" in the DataBox]]
SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.HelpersBadTensorFromBox",
                  "[Unit][DataStructures]") {
  ERROR_TEST();
  auto original_box =
      db::create<db::AddTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                             test_databox_tags::Tag2>,
                 db::AddComputeItemsTags<test_databox_tags::TagTensor>>(
          3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);

  std::pair<std::vector<std::string>, std::vector<double>> tag_tensor =
      get_tensor_from_box(original_box, "TagTensor__");
  static_cast<void>(tag_tensor);  // make sure compilers don't warn
}

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.Helpers",
                  "[Unit][DataStructures]") {
  auto original_box =
      db::create<db::AddTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                             test_databox_tags::Tag2>,
                 db::AddComputeItemsTags<test_databox_tags::TagTensor>>(
          3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);

  auto tag_tensor = get_tensor_from_box(original_box, "TagTensor");
  CHECK(tag_tensor.first == (std::vector<std::string>{"t"s, "x"s, "y"s, "z"s}));
  CHECK(tag_tensor.second[0] == 7.82);
  CHECK(tag_tensor.second[1] == 8.0);
  CHECK(tag_tensor.second[2] == 3.0);
  CHECK(tag_tensor.second[3] == 9.0);
  //  auto grid_coords_norm = get_tensor_norm_from_box(
  //      original_box, std::make_pair("GridCoordinates"s, TypeOfNorm::Max));
  //  CHECK(grid_coords_norm == decltype(grid_coords_norm){std::make_pair(
  //                                "x"s, std::make_pair(0.5, 3_st))});
}

// Test the tags
namespace {

auto get_vector() { return tnsr::I<DataVector, 3, Frame::Grid>(5_st,2.0); }

struct Var1 : db::ComputeItemTag {
  static constexpr db::DataBoxString_t label = "Var1";
  static constexpr auto function = get_vector;
  using argument_tags = typelist<>;
};

struct Var2 : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString_t label = "Var2";
};

using two_vars = typelist<Var1, Var2>;
using vector_only = typelist<Var1>;
using scalar_only = typelist<Var2>;

static_assert(
    cpp17::is_same_v<
        tmpl::back<db::wrap_tags_in<Tags::d, scalar_only, tmpl::size_t<2>,
                                    Frame::Grid>>::type,
        tnsr::i<DataVector, 2, Frame::Grid>>,
    "Failed db::wrap_tags_in scalar_only");

static_assert(
    cpp17::is_same_v<
        tmpl::back<db::wrap_tags_in<Tags::d, vector_only, tmpl::size_t<3>,
                                    Frame::Grid>>::type,
        tnsr::iJ<DataVector, 3, Frame::Grid>>,
    "Failed db::wrap_tags_in vector_only");

static_assert(
    cpp17::is_same_v<
        tmpl::back<db::wrap_tags_in<Tags::d, two_vars, tmpl::size_t<2>,
                                    Frame::Grid>>::type,
        tnsr::i<DataVector, 2, Frame::Grid>>,
    "Failed db::wrap_tags_in two_vars scalar");

static_assert(
    cpp17::is_same_v<
        tmpl::front<db::wrap_tags_in<Tags::d, two_vars, tmpl::size_t<3>,
                                    Frame::Grid>>::type,
        tnsr::iJ<DataVector, 3, Frame::Grid>>,
    "Failed db::wrap_tags_in two_vars vector");
}  // namespace
