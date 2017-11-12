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
  static constexpr db::DataBoxString_t label = "Tag0";
};
/// [databox_tag_example]
struct Tag1 : db::DataBoxTag {
  using type = std::vector<double>;
  static constexpr db::DataBoxString_t label = "Tag1";
};
struct Tag2 : db::DataBoxTag {
  using type = std::string;
  static constexpr db::DataBoxString_t label = "Tag2";
};
struct Tag3 : db::DataBoxTag {
  using type = std::string;
  static constexpr db::DataBoxString_t label = "Tag3";
};

/// [databox_compute_item_tag_example]
struct ComputeTag0 : db::ComputeItemTag {
  static constexpr db::DataBoxString_t label = "ComputeTag0";
  static constexpr auto function = multiply_by_two;
  using argument_tags = typelist<Tag0>;
};
/// [databox_compute_item_tag_example]
struct ComputeTag1 : db::ComputeItemTag {
  static constexpr db::DataBoxString_t label = "ComputeTag1";
  static constexpr auto function = append_word;
  using argument_tags = typelist<Tag2, ComputeTag0>;
};

struct TagTensor : db::ComputeItemTag {
  static constexpr db::DataBoxString_t label = "TagTensor";
  static constexpr auto function = get_tensor;
  using argument_tags = typelist<>;
};

/// [compute_item_tag_function]
struct ComputeLambda0 : db::ComputeItemTag {
  static constexpr db::DataBoxString_t label = "ComputeLambda0";
  static constexpr double function(const double a) { return 3.0 * a; }
  using argument_tags = typelist<Tag0>;
};
/// [compute_item_tag_function]

/// [compute_item_tag_no_tags]
struct ComputeLambda1 : db::ComputeItemTag {
  static constexpr db::DataBoxString_t label = "ComputeLambda1";
  static constexpr double function() { return 7.0; }
  using argument_tags = typelist<>;
};
/// [compute_item_tag_no_tags]

/// [databox_prefix_tag_example]
template <typename Tag>
struct TagPrefix : db::DataBoxPrefix {
  using type = typename Tag::type;
  using tag = Tag;
  static constexpr db::DataBoxString_t label = "TagPrefix";
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

static_assert(db::detail::tag_has_label<test_databox_tags::Tag0>::value,
              "Failed testing db::tag_has_label");
static_assert(db::detail::tag_has_label<test_databox_tags::TagTensor>::value,
              "Failed testing db::tag_has_label");
}  // namespace

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

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.mutate",
                  "[Unit][DataStructures]") {
  auto original_box =
      db::create<db::AddTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                             test_databox_tags::Tag2>,
                 db::AddComputeItemsTags<test_databox_tags::ComputeTag0,
                                         test_databox_tags::ComputeTag1>>(
          3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
  CHECK(approx(db::get<test_databox_tags::ComputeTag0>(original_box)) ==
        3.14 * 2.0);
  /// [databox_mutate_example]
  db::mutate<test_databox_tags::Tag0, test_databox_tags::Tag1>(
      original_box,
      [](double& tag0, std::vector<double>& tag1, const double& compute_tag0) {
        CHECK(6.28 == compute_tag0);
        tag0 = 10.32;
        tag1[0] = 837.2;
      },
      db::get<test_databox_tags::ComputeTag0>(original_box));
  CHECK(10.32 == db::get<test_databox_tags::Tag0>(original_box));
  CHECK(837.2 == db::get<test_databox_tags::Tag1>(original_box)[0]);
  /// [databox_mutate_example]
  CHECK(approx(db::get<test_databox_tags::ComputeTag0>(original_box)) ==
        10.32 * 2.0);
}

// [[OutputRegex, Unable to retrieve a \(compute\) item 'ComputeTag0' from the
// DataBox from within a call to mutate. You must pass these either through the
// capture list of the lambda or the constructor of a class, this restriction
// exists to avoid complexity]]
SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.mutate_locked_get",
                  "[Unit][DataStructures]") {
  ERROR_TEST();
  auto original_box =
      db::create<db::AddTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                             test_databox_tags::Tag2>,
                 db::AddComputeItemsTags<test_databox_tags::ComputeTag0,
                                         test_databox_tags::ComputeTag1>>(
          3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
  db::mutate<test_databox_tags::Tag0, test_databox_tags::Tag1>(
      original_box, [&original_box](double& tag0, std::vector<double>& tag1) {
        const auto& compute_tag0 =
            db::get<test_databox_tags::ComputeTag0>(original_box);
        tag0 = 10.32;
        tag1[0] = 837.2;
      });
}

// [[OutputRegex, Unable to retrieve a \(compute\) item 'ComputeTag0' from the
// DataBox from within a call to mutate. You must pass these either through the
// capture list of the lambda or the constructor of a class, this restriction
// exists to avoid complexity]]
SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.mutate_locked_get_lazy",
                  "[Unit][DataStructures]") {
  ERROR_TEST();
  auto original_box =
      db::create<db::AddTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                             test_databox_tags::Tag2>,
                 db::AddComputeItemsTags<test_databox_tags::ComputeTag0,
                                         test_databox_tags::ComputeTag1>>(
          3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
  db::mutate<test_databox_tags::Tag0, test_databox_tags::Tag1>(
      original_box,
      [&original_box](double& /*tag0*/, std::vector<double>& /*tag1*/) {
        const auto& compute_tag0 =
            original_box.template get_lazy<test_databox_tags::ComputeTag0>();
      });
}

// [[OutputRegex, Unable to mutate a DataBox that is already being mutated. This
// error occurs when mutating a DataBox from inside the invokable passed to the
// mutate function]]
SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.mutate_locked_mutate",
                  "[Unit][DataStructures]") {
  ERROR_TEST();
  auto original_box =
      db::create<db::AddTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                             test_databox_tags::Tag2>,
                 db::AddComputeItemsTags<test_databox_tags::ComputeTag0,
                                         test_databox_tags::ComputeTag1>>(
          3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
  db::mutate<test_databox_tags::Tag0>(
      original_box, [&original_box](double& /*unused*/) {
        db::mutate<test_databox_tags::Tag1>(
            original_box, [](std::vector<double>& tag1) { tag1[0] = 10.0; });
      });
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

auto get_vector() { return tnsr::I<DataVector, 3, Frame::Grid>(5_st, 2.0); }

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

static_assert(cpp17::is_same_v<
                  tmpl::back<db::wrap_tags_in<
                      Tags::d, two_vars, tmpl::size_t<2>, Frame::Grid>>::type,
                  tnsr::i<DataVector, 2, Frame::Grid>>,
              "Failed db::wrap_tags_in two_vars scalar");

static_assert(cpp17::is_same_v<
                  tmpl::front<db::wrap_tags_in<
                      Tags::d, two_vars, tmpl::size_t<3>, Frame::Grid>>::type,
                  tnsr::iJ<DataVector, 3, Frame::Grid>>,
              "Failed db::wrap_tags_in two_vars vector");
}  // namespace

namespace test_databox_tags {
struct ScalarTag : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString_t label = "ScalarTag";
};
struct VectorTag : db::DataBoxTag {
  using type = tnsr::I<DataVector, 3>;
  static constexpr db::DataBoxString_t label = "VectorTag";
};
struct ScalarTag2 : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString_t label = "ScalarTag2";
};
struct VectorTag2 : db::DataBoxTag {
  using type = tnsr::I<DataVector, 3>;
  static constexpr db::DataBoxString_t label = "VectorTag2";
};
struct ScalarTag3 : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString_t label = "ScalarTag3";
};
struct VectorTag3 : db::DataBoxTag {
  using type = tnsr::I<DataVector, 3>;
  static constexpr db::DataBoxString_t label = "VectorTag3";
};
struct ScalarTag4 : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString_t label = "ScalarTag4";
};
struct VectorTag4 : db::DataBoxTag {
  using type = tnsr::I<DataVector, 3>;
  static constexpr db::DataBoxString_t label = "VectorTag4";
};
}  // namespace test_databox_tags

namespace {
auto multiply_scalar_by_two(const Scalar<DataVector>& scalar) {
  Variables<
      tmpl::list<test_databox_tags::ScalarTag2, test_databox_tags::VectorTag2>>
      vars{scalar.begin()->size(), 2.0};
  get<test_databox_tags::ScalarTag2>(vars).get() = scalar.get() * 2.0;
  return vars;
}

auto multiply_scalar_by_four(const Scalar<DataVector>& scalar) {
  return Scalar<DataVector>(scalar.get() * 4.0);
}

auto multiply_scalar_by_three(const Scalar<DataVector>& scalar) {
  return Scalar<DataVector>(scalar.get() * 3.0);
}

auto divide_scalar_by_three(const Scalar<DataVector>& scalar) {
  return Scalar<DataVector>(scalar.get() / 3.0);
}

auto divide_scalar_by_two(const Scalar<DataVector>& scalar) {
  Variables<
      tmpl::list<test_databox_tags::VectorTag3, test_databox_tags::ScalarTag3>>
      vars{scalar.begin()->size(), 10.0};
  get<test_databox_tags::ScalarTag3>(vars).get() = scalar.get() / 2.0;
  return vars;
}

auto multiply_variables_by_two(
    const Variables<tmpl::list<test_databox_tags::ScalarTag,
                               test_databox_tags::VectorTag>>& vars) {
  Variables<
      tmpl::list<test_databox_tags::ScalarTag4, test_databox_tags::VectorTag4>>
      out_vars(vars.number_of_grid_points(), 2.0);
  get<test_databox_tags::ScalarTag4>(out_vars).get() *=
      get<test_databox_tags::ScalarTag>(vars).get();
  get<test_databox_tags::VectorTag4>(out_vars).get<0>() *=
      get<test_databox_tags::VectorTag>(vars).get<0>();
  get<test_databox_tags::VectorTag4>(out_vars).get<1>() *=
      get<test_databox_tags::VectorTag>(vars).get<1>();
  get<test_databox_tags::VectorTag4>(out_vars).get<2>() *=
      get<test_databox_tags::VectorTag>(vars).get<2>();
  return out_vars;
}
}  // namespace

namespace test_databox_tags {
struct MultiplyScalarByTwo : db::ComputeItemTag {
  using variables_tags =
      tmpl::list<test_databox_tags::ScalarTag2, test_databox_tags::VectorTag2>;
  static constexpr db::DataBoxString_t label = "MultiplyScalarByTwo";
  static constexpr auto function = multiply_scalar_by_two;
  using argument_tags = typelist<test_databox_tags::ScalarTag>;
};

struct MultiplyScalarByFour : db::ComputeItemTag {
  static constexpr db::DataBoxString_t label = "MultiplyScalarByFour";
  static constexpr auto function = multiply_scalar_by_four;
  using argument_tags = typelist<test_databox_tags::ScalarTag2>;
};

struct MultiplyScalarByThree : db::ComputeItemTag {
  static constexpr db::DataBoxString_t label = "MultiplyScalarByThree";
  static constexpr auto function = multiply_scalar_by_three;
  using argument_tags = typelist<test_databox_tags::MultiplyScalarByFour>;
};

struct DivideScalarByThree : db::ComputeItemTag {
  static constexpr db::DataBoxString_t label = "DivideScalarByThree";
  static constexpr auto function = divide_scalar_by_three;
  using argument_tags = typelist<test_databox_tags::MultiplyScalarByThree>;
};

struct DivideScalarByTwo : db::ComputeItemTag {
  static constexpr db::DataBoxString_t label = "DivideScalarByTwo";
  static constexpr auto function = divide_scalar_by_two;
  using argument_tags = typelist<test_databox_tags::DivideScalarByThree>;
};

struct MultiplyVariablesByTwo : db::ComputeItemTag {
  static constexpr db::DataBoxString_t label = "MultiplyVariablesByTwo";
  static constexpr auto function = multiply_variables_by_two;
  using argument_tags = typelist<Tags::Variables<
      tmpl::list<test_databox_tags::ScalarTag, test_databox_tags::VectorTag>>>;
};
}  // namespace test_databox_tags

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.Variables",
                  "[Unit][DataStructures]") {
  auto box = db::create<
      db::AddTags<Tags::Variables<tmpl::list<test_databox_tags::ScalarTag,
                                             test_databox_tags::VectorTag>>>,
      db::AddComputeItemsTags<test_databox_tags::MultiplyScalarByTwo,
                              test_databox_tags::MultiplyScalarByFour,
                              test_databox_tags::MultiplyScalarByThree,
                              test_databox_tags::DivideScalarByThree,
                              test_databox_tags::DivideScalarByTwo,
                              test_databox_tags::MultiplyVariablesByTwo>>(
      Variables<tmpl::list<test_databox_tags::ScalarTag,
                           test_databox_tags::VectorTag>>(2, 3.));
  CHECK(db::get<test_databox_tags::ScalarTag>(box) ==
        Scalar<DataVector>(DataVector(2, 3.)));
  CHECK(db::get<test_databox_tags::VectorTag>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 3.))));
  CHECK(db::get<test_databox_tags::ScalarTag2>(box) ==
        Scalar<DataVector>(DataVector(2, 6.)));
  CHECK(db::get<test_databox_tags::VectorTag2>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 2.))));
  CHECK(db::get<test_databox_tags::MultiplyScalarByFour>(box) ==
        Scalar<DataVector>(DataVector(2, 24.)));
  CHECK(db::get<test_databox_tags::MultiplyScalarByThree>(box) ==
        Scalar<DataVector>(DataVector(2, 72.)));
  CHECK(db::get<test_databox_tags::DivideScalarByThree>(box) ==
        Scalar<DataVector>(DataVector(2, 24.)));
  CHECK(db::get<test_databox_tags::ScalarTag3>(box) ==
        Scalar<DataVector>(DataVector(2, 12.)));
  CHECK(db::get<test_databox_tags::VectorTag3>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 10.))));
  CHECK(db::get<test_databox_tags::ScalarTag4>(box) ==
        Scalar<DataVector>(DataVector(2, 6.)));
  CHECK(db::get<test_databox_tags::VectorTag4>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 6.))));
  {
    const auto& vars = db::get<test_databox_tags::MultiplyVariablesByTwo>(box);
    CHECK(get<test_databox_tags::ScalarTag4>(vars) ==
          Scalar<DataVector>(DataVector(2, 6.)));
    CHECK(get<test_databox_tags::VectorTag4>(vars) ==
          (tnsr::I<DataVector, 3>(DataVector(2, 6.))));
  }

  db::mutate<test_databox_tags::ScalarTag>(
      box, [](Scalar<DataVector>& scalar) { scalar.get() = 4.0; });

  CHECK(db::get<test_databox_tags::ScalarTag>(box) ==
        Scalar<DataVector>(DataVector(2, 4.)));
  CHECK(db::get<test_databox_tags::VectorTag>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 3.))));
  CHECK(db::get<test_databox_tags::ScalarTag2>(box) ==
        Scalar<DataVector>(DataVector(2, 8.)));
  CHECK(db::get<test_databox_tags::VectorTag2>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 2.))));
  CHECK(db::get<test_databox_tags::MultiplyScalarByFour>(box) ==
        Scalar<DataVector>(DataVector(2, 32.)));
  CHECK(db::get<test_databox_tags::MultiplyScalarByThree>(box) ==
        Scalar<DataVector>(DataVector(2, 96.)));
  CHECK(db::get<test_databox_tags::DivideScalarByThree>(box) ==
        Scalar<DataVector>(DataVector(2, 32.)));
  CHECK(db::get<test_databox_tags::ScalarTag3>(box) ==
        Scalar<DataVector>(DataVector(2, 16.)));
  CHECK(db::get<test_databox_tags::VectorTag3>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 10.))));
  CHECK(db::get<test_databox_tags::ScalarTag4>(box) ==
        Scalar<DataVector>(DataVector(2, 8.)));
  CHECK(db::get<test_databox_tags::VectorTag4>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 6.))));
  {
    const auto& vars = db::get<test_databox_tags::MultiplyVariablesByTwo>(box);
    CHECK(get<test_databox_tags::ScalarTag4>(vars) ==
          Scalar<DataVector>(DataVector(2, 8.)));
    CHECK(get<test_databox_tags::VectorTag4>(vars) ==
          (tnsr::I<DataVector, 3>(DataVector(2, 6.))));
  }

  db::mutate<Tags::Variables<
      tmpl::list<test_databox_tags::ScalarTag, test_databox_tags::VectorTag>>>(
      box, [](auto& vars) {
        const auto size = vars.number_of_grid_points();
        get<test_databox_tags::ScalarTag>(vars).get() = 6.0;
      });

  CHECK(db::get<test_databox_tags::ScalarTag>(box) ==
        Scalar<DataVector>(DataVector(2, 6.)));
  CHECK(db::get<test_databox_tags::VectorTag>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 3.))));
  CHECK(db::get<test_databox_tags::ScalarTag2>(box) ==
        Scalar<DataVector>(DataVector(2, 12.)));
  CHECK(db::get<test_databox_tags::VectorTag2>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 2.))));
  CHECK(db::get<test_databox_tags::MultiplyScalarByFour>(box) ==
        Scalar<DataVector>(DataVector(2, 48.)));
  CHECK(db::get<test_databox_tags::MultiplyScalarByThree>(box) ==
        Scalar<DataVector>(DataVector(2, 144.)));
  CHECK(db::get<test_databox_tags::DivideScalarByThree>(box) ==
        Scalar<DataVector>(DataVector(2, 48.)));
  CHECK(db::get<test_databox_tags::ScalarTag3>(box) ==
        Scalar<DataVector>(DataVector(2, 24.)));
  CHECK(db::get<test_databox_tags::VectorTag3>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 10.))));
  CHECK(db::get<test_databox_tags::ScalarTag4>(box) ==
        Scalar<DataVector>(DataVector(2, 12.)));
  CHECK(db::get<test_databox_tags::VectorTag4>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 6.))));
  {
    const auto& vars = db::get<test_databox_tags::MultiplyVariablesByTwo>(box);
    CHECK(get<test_databox_tags::ScalarTag4>(vars) ==
          Scalar<DataVector>(DataVector(2, 12.)));
    CHECK(get<test_databox_tags::VectorTag4>(vars) ==
          (tnsr::I<DataVector, 3>(DataVector(2, 6.))));
  }

  db::mutate<Tags::Variables<
      tmpl::list<test_databox_tags::ScalarTag, test_databox_tags::VectorTag>>>(
      box, [](auto& vars) {
        const auto size = vars.number_of_grid_points();
        get<test_databox_tags::ScalarTag>(vars).get() = 4.0;
        get<test_databox_tags::VectorTag>(vars) =
            tnsr::I<DataVector, 3>(DataVector(2, 6.));
      });

  CHECK(db::get<test_databox_tags::ScalarTag>(box) ==
        Scalar<DataVector>(DataVector(2, 4.)));
  CHECK(db::get<test_databox_tags::VectorTag>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 6.))));
  CHECK(db::get<test_databox_tags::ScalarTag2>(box) ==
        Scalar<DataVector>(DataVector(2, 8.)));
  CHECK(db::get<test_databox_tags::VectorTag2>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 2.))));
  CHECK(db::get<test_databox_tags::MultiplyScalarByFour>(box) ==
        Scalar<DataVector>(DataVector(2, 32.)));
  CHECK(db::get<test_databox_tags::MultiplyScalarByThree>(box) ==
        Scalar<DataVector>(DataVector(2, 96.)));
  CHECK(db::get<test_databox_tags::DivideScalarByThree>(box) ==
        Scalar<DataVector>(DataVector(2, 32.)));
  CHECK(db::get<test_databox_tags::ScalarTag3>(box) ==
        Scalar<DataVector>(DataVector(2, 16.)));
  CHECK(db::get<test_databox_tags::VectorTag3>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 10.))));
  CHECK(db::get<test_databox_tags::ScalarTag4>(box) ==
        Scalar<DataVector>(DataVector(2, 8.)));
  CHECK(db::get<test_databox_tags::VectorTag4>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 12.))));
  {
    const auto& vars = db::get<test_databox_tags::MultiplyVariablesByTwo>(box);
    CHECK(get<test_databox_tags::ScalarTag4>(vars) ==
          Scalar<DataVector>(DataVector(2, 8.)));
    CHECK(get<test_databox_tags::VectorTag4>(vars) ==
          (tnsr::I<DataVector, 3>(DataVector(2, 12.))));
  }
}

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.reset_compute_items",
                  "[Unit][DataStructures]") {
  auto box = db::create<
      db::AddTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                  test_databox_tags::Tag2,
                  Tags::Variables<tmpl::list<test_databox_tags::ScalarTag,
                                             test_databox_tags::VectorTag>>>,
      db::AddComputeItemsTags<test_databox_tags::ComputeTag0,
                              test_databox_tags::ComputeTag1,
                              test_databox_tags::MultiplyScalarByTwo,
                              test_databox_tags::MultiplyVariablesByTwo>>(
      3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s,
      Variables<tmpl::list<test_databox_tags::ScalarTag,
                           test_databox_tags::VectorTag>>(2, 3.));
  CHECK(approx(db::get<test_databox_tags::ComputeTag0>(box)) == 3.14 * 2.0);
  CHECK(db::get<test_databox_tags::ScalarTag>(box) ==
        Scalar<DataVector>(DataVector(2, 3.)));
  CHECK(db::get<test_databox_tags::VectorTag>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 3.))));
  CHECK(db::get<test_databox_tags::ScalarTag2>(box) ==
        Scalar<DataVector>(DataVector(2, 6.)));
  CHECK(db::get<test_databox_tags::VectorTag2>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 2.))));
  CHECK(db::get<test_databox_tags::ScalarTag4>(box) ==
        Scalar<DataVector>(DataVector(2, 6.)));
  CHECK(db::get<test_databox_tags::VectorTag4>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 6.))));
  {
    const auto& vars = db::get<test_databox_tags::MultiplyVariablesByTwo>(box);
    CHECK(get<test_databox_tags::ScalarTag4>(vars) ==
          Scalar<DataVector>(DataVector(2, 6.)));
    CHECK(get<test_databox_tags::VectorTag4>(vars) ==
          (tnsr::I<DataVector, 3>(DataVector(2, 6.))));
  }

  auto box2 = db::create_from<
      db::RemoveTags<test_databox_tags::Tag0,
                     Tags::Variables<tmpl::list<test_databox_tags::ScalarTag,
                                                test_databox_tags::VectorTag>>>,
      db::AddTags<test_databox_tags::Tag0,
                  Tags::Variables<tmpl::list<test_databox_tags::ScalarTag,
                                             test_databox_tags::VectorTag>>>>(
      box, 3.84,
      Variables<tmpl::list<test_databox_tags::ScalarTag,
                           test_databox_tags::VectorTag>>(4, 8.0));

  CHECK(approx(db::get<test_databox_tags::ComputeTag0>(box2)) == 3.84 * 2.0);
  CHECK(db::get<test_databox_tags::ScalarTag>(box2) ==
        Scalar<DataVector>(DataVector(4, 8.)));
  CHECK(db::get<test_databox_tags::VectorTag>(box2) ==
        (tnsr::I<DataVector, 3>(DataVector(4, 8.))));
  CHECK(db::get<test_databox_tags::ScalarTag2>(box2) ==
        Scalar<DataVector>(DataVector(4, 16.)));
  CHECK(db::get<test_databox_tags::VectorTag2>(box2) ==
        (tnsr::I<DataVector, 3>(DataVector(4, 2.))));
  CHECK(db::get<test_databox_tags::ScalarTag4>(box2) ==
        Scalar<DataVector>(DataVector(4, 16.)));
  CHECK(db::get<test_databox_tags::VectorTag4>(box2) ==
        (tnsr::I<DataVector, 3>(DataVector(4, 16.))));
  {
    const auto& vars = db::get<test_databox_tags::MultiplyVariablesByTwo>(box2);
    CHECK(get<test_databox_tags::ScalarTag4>(vars) ==
          Scalar<DataVector>(DataVector(4, 16.)));
    CHECK(get<test_databox_tags::VectorTag4>(vars) ==
          (tnsr::I<DataVector, 3>(DataVector(4, 16.))));
  }
}


namespace {
/// [mutate_apply_apply_struct_example]
struct test_databox_mutate_apply {
  static void apply(const gsl::not_null<Scalar<DataVector>*> scalar,
                    const gsl::not_null<tnsr::I<DataVector, 3>*> vector,
                    const std::string& tag2) {
    scalar->get() *= 2.0;
    vector->template get<0>() *= 3.0;
    vector->template get<1>() *= 4.0;
    vector->template get<2>() *= 5.0;
    CHECK(tag2 == "My Sample String"s);
  }
};
/// [mutate_apply_apply_struct_example]
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.mutate_apply",
                  "[Unit][DataStructures]") {
  auto box = db::create<
      db::AddTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                  test_databox_tags::Tag2,
                  Tags::Variables<tmpl::list<test_databox_tags::ScalarTag,
                                             test_databox_tags::VectorTag>>>,
      db::AddComputeItemsTags<test_databox_tags::ComputeTag0,
                              test_databox_tags::ComputeTag1,
                              test_databox_tags::MultiplyScalarByTwo>>(
      3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s,
      Variables<tmpl::list<test_databox_tags::ScalarTag,
                           test_databox_tags::VectorTag>>(2, 3.));
  CHECK(approx(db::get<test_databox_tags::ComputeTag0>(box)) == 3.14 * 2.0);
  CHECK(db::get<test_databox_tags::ScalarTag>(box) ==
        Scalar<DataVector>(DataVector(2, 3.)));
  CHECK(db::get<test_databox_tags::VectorTag>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 3.))));
  CHECK(db::get<test_databox_tags::ScalarTag2>(box) ==
        Scalar<DataVector>(DataVector(2, 6.)));
  CHECK(db::get<test_databox_tags::VectorTag2>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 2.))));

  /// [mutate_apply_example]
  db::mutate_apply<
      tmpl::list<test_databox_tags::ScalarTag, test_databox_tags::VectorTag>,
      tmpl::list<>>(test_databox_mutate_apply{}, box,
                    db::get<test_databox_tags::Tag2>(box));
  /// [mutate_apply_example]
  CHECK(approx(db::get<test_databox_tags::ComputeTag0>(box)) == 3.14 * 2.0);
  CHECK(db::get<test_databox_tags::ScalarTag>(box) ==
        Scalar<DataVector>(DataVector(2, 6.)));
  CHECK(db::get<test_databox_tags::VectorTag>(box) ==
        (tnsr::I<DataVector, 3>{
            {{DataVector(2, 9.), DataVector(2, 12.), DataVector(2, 15.)}}}));
  CHECK(db::get<test_databox_tags::ScalarTag2>(box) ==
        Scalar<DataVector>(DataVector(2, 12.)));
  CHECK(db::get<test_databox_tags::VectorTag2>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 2.))));
  /// [mutate_apply_apply_example]
  db::mutate_apply<
      tmpl::list<test_databox_tags::ScalarTag, test_databox_tags::VectorTag>,
      tmpl::list<test_databox_tags::Tag2>>(
      [](const gsl::not_null<Scalar<DataVector>*> scalar,
         const gsl::not_null<tnsr::I<DataVector, 3>*> vector,
         const std::string& tag2) {
        scalar->get() *= 2.0;
        vector->template get<0>() *= 3.0;
        vector->template get<1>() *= 4.0;
        vector->template get<2>() *= 5.0;
        CHECK(tag2 == "My Sample String"s);
      },
      box);
  /// [mutate_apply_apply_example]
  CHECK(approx(db::get<test_databox_tags::ComputeTag0>(box)) == 3.14 * 2.0);
  CHECK(db::get<test_databox_tags::ScalarTag>(box) ==
        Scalar<DataVector>(DataVector(2, 12.)));
  CHECK(db::get<test_databox_tags::VectorTag>(box) ==
        (tnsr::I<DataVector, 3>{
            {{DataVector(2, 27.), DataVector(2, 48.), DataVector(2, 75.)}}}));
  CHECK(db::get<test_databox_tags::ScalarTag2>(box) ==
        Scalar<DataVector>(DataVector(2, 24.)));
  CHECK(db::get<test_databox_tags::VectorTag2>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 2.))));

  db::mutate_apply<
      tmpl::list<Tags::Variables<tmpl::list<test_databox_tags::ScalarTag,
                                            test_databox_tags::VectorTag>>>,
      tmpl::list<test_databox_tags::Tag2>>(
      [](const gsl::not_null<Variables<tmpl::list<
             test_databox_tags::ScalarTag, test_databox_tags::VectorTag>>*>
             vars,
         const std::string& tag2) {
        get<test_databox_tags::ScalarTag>(*vars).get() *= 2.0;
        get<test_databox_tags::VectorTag>(*vars).template get<0>() *= 3.0;
        get<test_databox_tags::VectorTag>(*vars).template get<1>() *= 4.0;
        get<test_databox_tags::VectorTag>(*vars).template get<2>() *= 5.0;
        CHECK(tag2 == "My Sample String"s);
      },
      box);

  CHECK(approx(db::get<test_databox_tags::ComputeTag0>(box)) == 3.14 * 2.0);
  CHECK(db::get<test_databox_tags::ScalarTag>(box) ==
        Scalar<DataVector>(DataVector(2, 24.)));
  CHECK(db::get<test_databox_tags::VectorTag>(box) ==
        (tnsr::I<DataVector, 3>{
            {{DataVector(2, 81.), DataVector(2, 192.), DataVector(2, 375.)}}}));
  CHECK(db::get<test_databox_tags::ScalarTag2>(box) ==
        Scalar<DataVector>(DataVector(2, 48.)));
  CHECK(db::get<test_databox_tags::VectorTag2>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 2.))));
}
