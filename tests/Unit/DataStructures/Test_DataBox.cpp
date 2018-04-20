// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <ostream>
#include <pup.h>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxHelpers.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/DataOnSlice.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
template <typename TagsList>
class Variables;
template <typename X, typename Symm, typename IndexList>
class Tensor;

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
  static constexpr db::DataBoxString label = "Tag0";
};
/// [databox_tag_example]
struct Tag1 : db::DataBoxTag {
  using type = std::vector<double>;
  static constexpr db::DataBoxString label = "Tag1";
};
struct Tag2 : db::DataBoxTag {
  using type = std::string;
  static constexpr db::DataBoxString label = "Tag2";
};
struct Tag3 : db::DataBoxTag {
  using type = std::string;
  static constexpr db::DataBoxString label = "Tag3";
};

/// [databox_compute_item_tag_example]
struct ComputeTag0 : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "ComputeTag0";
  static constexpr auto function = multiply_by_two;
  using argument_tags = tmpl::list<Tag0>;
};
/// [databox_compute_item_tag_example]
struct ComputeTag1 : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "ComputeTag1";
  static constexpr auto function = append_word;
  using argument_tags = tmpl::list<Tag2, ComputeTag0>;
};

struct TagTensor : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "TagTensor";
  static constexpr auto function = get_tensor;
  using argument_tags = tmpl::list<>;
};

/// [compute_item_tag_function]
struct ComputeLambda0 : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "ComputeLambda0";
  static constexpr double function(const double a) { return 3.0 * a; }
  using argument_tags = tmpl::list<Tag0>;
};
/// [compute_item_tag_function]

/// [compute_item_tag_no_tags]
struct ComputeLambda1 : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "ComputeLambda1";
  static constexpr double function() { return 7.0; }
  using argument_tags = tmpl::list<>;
};
/// [compute_item_tag_no_tags]

/// [databox_prefix_tag_example]
template <typename Tag>
struct TagPrefix : db::DataBoxPrefix {
  using type = typename Tag::type;
  using tag = Tag;
  static constexpr db::DataBoxString label = "TagPrefix";
};
/// [databox_prefix_tag_example]
}  // namespace test_databox_tags

namespace {
using Box_t = db::DataBox<tmpl::list<
    test_databox_tags::Tag0, test_databox_tags::Tag1, test_databox_tags::Tag2,
    test_databox_tags::TagPrefix<test_databox_tags::Tag0>,
    test_databox_tags::ComputeTag0, test_databox_tags::ComputeTag1>>;

static_assert(
    std::is_same<
        decltype(
            db::create_from<db::RemoveTags<test_databox_tags::Tag1>>(Box_t{})),
        db::DataBox<
            tmpl::list<test_databox_tags::Tag0, test_databox_tags::Tag2,
                       test_databox_tags::TagPrefix<test_databox_tags::Tag0>,
                       test_databox_tags::ComputeTag0,
                       test_databox_tags::ComputeTag1>>>::value,
    "Failed testing removal of item");

static_assert(
    std::is_same<
        decltype(db::create_from<
                 db::RemoveTags<test_databox_tags::ComputeTag1>>(Box_t{})),
        db::DataBox<
            tmpl::list<test_databox_tags::Tag0, test_databox_tags::Tag1,
                       test_databox_tags::Tag2,
                       test_databox_tags::TagPrefix<test_databox_tags::Tag0>,
                       test_databox_tags::ComputeTag0>>>::value,
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
  auto original_box = db::create<
      db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                        test_databox_tags::Tag2>,
      db::AddComputeTags<test_databox_tags::ComputeTag0,
                         test_databox_tags::ComputeTag1,
                         test_databox_tags::ComputeLambda0,
                         test_databox_tags::ComputeLambda1>>(
      3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
  /// [create_databox]
  static_assert(
      std::is_same<
          decltype(original_box),
          db::DataBox<db::databox_detail::dependent_items<tmpl::list<
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
    auto box = db::create_from<db::RemoveTags<>,
                               db::AddSimpleTags<test_databox_tags::Tag3>>(
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
    auto simple_box = db::create<
        db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                          test_databox_tags::Tag2>>(
        3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
    auto box =
        db::create_from<db::RemoveTags<>, db::AddSimpleTags<>,
                        db::AddComputeTags<test_databox_tags::ComputeTag0>>(
            simple_box);
    /// [create_from_add_compute_item]
    CHECK(db::get<test_databox_tags::Tag2>(box) == "My Sample String"s);
    // Check retrieving compute item result
    CHECK(db::get<test_databox_tags::ComputeTag0>(box) == 6.28);
  }
  {
    auto simple_box = db::create<
        db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                          test_databox_tags::Tag2>>(
        3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
    auto box =
        db::create_from<db::RemoveTags<>,
                        db::AddSimpleTags<test_databox_tags::Tag3>,
                        db::AddComputeTags<test_databox_tags::ComputeTag0>>(
            simple_box, "Yet another test string"s);
    CHECK(db::get<test_databox_tags::Tag3>(box) == "Yet another test string"s);
    CHECK(db::get<test_databox_tags::Tag2>(box) == "My Sample String"s);
    // Check retrieving compute item result
    CHECK(db::get<test_databox_tags::ComputeTag0>(box) == 6.28);
  }
  {
    auto simple_box = db::create<
        db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                          test_databox_tags::Tag2>>(
        3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
    auto box =
        db::create_from<db::RemoveTags<test_databox_tags::Tag1>,
                        db::AddSimpleTags<test_databox_tags::Tag3>,
                        db::AddComputeTags<test_databox_tags::ComputeTag0>>(
            simple_box, "Yet another test string"s);
    CHECK(db::get<test_databox_tags::Tag3>(box) == "Yet another test string"s);
    CHECK(db::get<test_databox_tags::Tag2>(box) == "My Sample String"s);
    // Check retrieving compute item result
    CHECK(6.28 == db::get<test_databox_tags::ComputeTag0>(box));
  }
}

namespace ArgumentTypeTags {
struct NonCopyable : db::DataBoxTag {
  static constexpr db::DataBoxString label = "NonCopyable";
  using type = ::NonCopyable;
};
template <size_t N>
struct String : db::DataBoxTag {
  static constexpr db::DataBoxString label = "String";
  using type = std::string;
};
}  // namespace ArgumentTypeTags
SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.create_argument_types",
                  "[Unit][DataStructures]") {
  std::string mutable_string = "mutable";
  const std::string const_string = "const";
  std::string move_string = "move";
  const std::string const_move_string = "const move";
  // clang-tidy: std::move of a const variable
  auto box = db::create<db::AddSimpleTags<
      ArgumentTypeTags::NonCopyable, ArgumentTypeTags::String<0>,
      ArgumentTypeTags::String<1>, ArgumentTypeTags::String<2>,
      ArgumentTypeTags::String<3>>>(NonCopyable{}, mutable_string, const_string,
                                    std::move(move_string),
                                    std::move(const_move_string));  // NOLINT
  CHECK(mutable_string == "mutable");
  CHECK(const_string == "const");
  CHECK(db::get<ArgumentTypeTags::String<0>>(box) == "mutable");
  CHECK(db::get<ArgumentTypeTags::String<1>>(box) == "const");
  CHECK(db::get<ArgumentTypeTags::String<2>>(box) == "move");
  CHECK(db::get<ArgumentTypeTags::String<3>>(box) == "const move");
}

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.get_databox",
                  "[Unit][DataStructures]") {
  auto original_box = db::create<
      db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                        test_databox_tags::Tag2>,
      db::AddComputeTags<test_databox_tags::ComputeTag0,
                         test_databox_tags::ComputeTag1>>(
      3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
  CHECK(std::addressof(original_box) ==
        std::addressof(db::get<Tags::DataBox>(original_box)));
  /// [databox_self_tag_example]
  auto check_result_no_args = [](const auto& box) {
    CHECK(db::get<test_databox_tags::Tag2>(box) == "My Sample String"s);
    CHECK(db::get<test_databox_tags::ComputeTag1>(box) ==
          "My Sample String6.28"s);
  };
  db::apply<tmpl::list<Tags::DataBox>>(check_result_no_args, original_box);
  /// [databox_self_tag_example]
}

// [[OutputRegex, Unable to retrieve a \(compute\) item 'DataBox' from the
// DataBox from within a call to mutate. You must pass these either through the
// capture list of the lambda or the constructor of a class, this restriction
// exists to avoid complexity.]]
SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.get_databox_error",
                  "[Unit][DataStructures]") {
  ERROR_TEST();
  auto original_box = db::create<
      db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                        test_databox_tags::Tag2>,
      db::AddComputeTags<test_databox_tags::ComputeTag0,
                         test_databox_tags::ComputeTag1>>(
      3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
  CHECK(std::addressof(original_box) ==
        std::addressof(db::get<Tags::DataBox>(original_box)));
  db::mutate<test_databox_tags::Tag0>(
      original_box, [&original_box](double& /*tag0*/) {
        (void)db::get<Tags::DataBox>(original_box);
      });
}

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.mutate",
                  "[Unit][DataStructures]") {
  auto original_box = db::create<
      db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                        test_databox_tags::Tag2>,
      db::AddComputeTags<test_databox_tags::ComputeTag0,
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
  auto original_box = db::create<
      db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                        test_databox_tags::Tag2>,
      db::AddComputeTags<test_databox_tags::ComputeTag0,
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
  auto original_box = db::create<
      db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                        test_databox_tags::Tag2>,
      db::AddComputeTags<test_databox_tags::ComputeTag0,
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
  auto original_box = db::create<
      db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                        test_databox_tags::Tag2>,
      db::AddComputeTags<test_databox_tags::ComputeTag0,
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
      db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                        test_databox_tags::Tag2,
                        test_databox_tags::TagPrefix<test_databox_tags::Tag0>>,
      db::AddComputeTags<test_databox_tags::ComputeTag0,
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
  auto original_box = db::create<
      db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                        test_databox_tags::Tag2>,
      db::AddComputeTags<test_databox_tags::ComputeTag0,
                         test_databox_tags::ComputeTag1>>(
      3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
  static_cast<void>(db::get_item_from_box<double>(original_box, "time__"));
}

namespace {
struct NonCopyableFunctor {
  NonCopyableFunctor() = default;
  NonCopyableFunctor(const NonCopyableFunctor&) = delete;
  NonCopyableFunctor(NonCopyableFunctor&&) = delete;
  NonCopyableFunctor& operator=(const NonCopyableFunctor&) = delete;
  NonCopyableFunctor& operator=(NonCopyableFunctor&&) = delete;
  ~NonCopyableFunctor() = default;

  // The && before the function body requires the object to be an
  // rvalue for the method to be called.  This checks that the apply
  // functions correctly preserve the value category of the functor.
  template <typename... Args>
  void operator()(Args&&... /*unused*/) && {}
};
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.apply",
                  "[Unit][DataStructures]") {
  auto original_box = db::create<
      db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                        test_databox_tags::Tag2>,
      db::AddComputeTags<test_databox_tags::ComputeTag0,
                         test_databox_tags::ComputeTag1>>(
      3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
  auto check_result_no_args = [](const std::string& sample_string,
                                 const auto& computed_string) {
    CHECK(sample_string == "My Sample String"s);
    CHECK(computed_string == "My Sample String6.28"s);
  };
  db::apply<
      tmpl::list<test_databox_tags::Tag2, test_databox_tags::ComputeTag1>>(
      check_result_no_args, original_box);

  /// [apply_example]
  auto check_result_args = [](const std::string& sample_string,
                              const auto& computed_string, const auto& vector) {
    CHECK(sample_string == "My Sample String"s);
    CHECK(computed_string == "My Sample String6.28"s);
    CHECK(vector == (std::vector<double>{8.7, 93.2, 84.7}));
  };
  db::apply<
      tmpl::list<test_databox_tags::Tag2, test_databox_tags::ComputeTag1>>(
      check_result_args, original_box,
      db::get<test_databox_tags::Tag1>(original_box));
  /// [apply_example]

  db::apply<tmpl::list<>>(NonCopyableFunctor{}, original_box);
}

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.apply_with_box",
                  "[Unit][DataStructures]") {
  auto original_box = db::create<
      db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                        test_databox_tags::Tag2>,
      db::AddComputeTags<test_databox_tags::ComputeTag0,
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
      tmpl::list<test_databox_tags::Tag2, test_databox_tags::ComputeTag1>>(
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
      tmpl::list<test_databox_tags::Tag2, test_databox_tags::ComputeTag1>>(
      check_result_args, original_box, std::vector<int>{1, 4, 8});
  /// [apply_with_box_example]

  db::apply_with_box<tmpl::list<>>(NonCopyableFunctor{}, original_box);
}

// [[OutputRegex, Could not find the tag named "TagTensor__" in the DataBox]]
SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.HelpersBadTensorFromBox",
                  "[Unit][DataStructures]") {
  ERROR_TEST();
  auto original_box = db::create<
      db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                        test_databox_tags::Tag2>,
      db::AddComputeTags<test_databox_tags::TagTensor>>(
      3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);

  std::pair<std::vector<std::string>, std::vector<double>> tag_tensor =
      get_tensor_from_box(original_box, "TagTensor__");
  static_cast<void>(tag_tensor);  // make sure compilers don't warn
}

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.Helpers",
                  "[Unit][DataStructures]") {
  auto original_box = db::create<
      db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                        test_databox_tags::Tag2>,
      db::AddComputeTags<test_databox_tags::TagTensor>>(
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
  static constexpr db::DataBoxString label = "Var1";
  static constexpr auto function = get_vector;
  using argument_tags = tmpl::list<>;
};

struct Var2 : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString label = "Var2";
};

template <class Tag, class VolumeDim, class Frame>
struct PrefixTag0 : db::DataBoxPrefix {
  using type = TensorMetafunctions::prepend_spatial_index<
      db::item_type<Tag>, VolumeDim::value, UpLo::Lo, Frame>;
  using tag = Tag;
  static constexpr db::DataBoxString label = "PrefixTag0";
};

using two_vars = tmpl::list<Var1, Var2>;
using vector_only = tmpl::list<Var1>;
using scalar_only = tmpl::list<Var2>;

static_assert(
    cpp17::is_same_v<
        tmpl::back<db::wrap_tags_in<PrefixTag0, scalar_only, tmpl::size_t<2>,
                                    Frame::Grid>>::type,
        tnsr::i<DataVector, 2, Frame::Grid>>,
    "Failed db::wrap_tags_in scalar_only");

static_assert(
    cpp17::is_same_v<
        tmpl::back<db::wrap_tags_in<PrefixTag0, vector_only, tmpl::size_t<3>,
                                    Frame::Grid>>::type,
        tnsr::iJ<DataVector, 3, Frame::Grid>>,
    "Failed db::wrap_tags_in vector_only");

static_assert(
    cpp17::is_same_v<
        tmpl::back<db::wrap_tags_in<PrefixTag0, two_vars, tmpl::size_t<2>,
                                    Frame::Grid>>::type,
        tnsr::i<DataVector, 2, Frame::Grid>>,
    "Failed db::wrap_tags_in two_vars scalar");

static_assert(
    cpp17::is_same_v<
        tmpl::front<db::wrap_tags_in<PrefixTag0, two_vars, tmpl::size_t<3>,
                                     Frame::Grid>>::type,
        tnsr::iJ<DataVector, 3, Frame::Grid>>,
    "Failed db::wrap_tags_in two_vars vector");
}  // namespace

namespace test_databox_tags {
struct ScalarTag : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString label = "ScalarTag";
};
struct VectorTag : db::DataBoxTag {
  using type = tnsr::I<DataVector, 3>;
  static constexpr db::DataBoxString label = "VectorTag";
};
struct ScalarTag2 : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString label = "ScalarTag2";
};
struct VectorTag2 : db::DataBoxTag {
  using type = tnsr::I<DataVector, 3>;
  static constexpr db::DataBoxString label = "VectorTag2";
};
struct ScalarTag3 : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString label = "ScalarTag3";
};
struct VectorTag3 : db::DataBoxTag {
  using type = tnsr::I<DataVector, 3>;
  static constexpr db::DataBoxString label = "VectorTag3";
};
struct ScalarTag4 : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString label = "ScalarTag4";
};
struct VectorTag4 : db::DataBoxTag {
  using type = tnsr::I<DataVector, 3>;
  static constexpr db::DataBoxString label = "VectorTag4";
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
  get<0>(get<test_databox_tags::VectorTag4>(out_vars)) *=
      get<0>(get<test_databox_tags::VectorTag>(vars));
  get<1>(get<test_databox_tags::VectorTag4>(out_vars)) *=
      get<1>(get<test_databox_tags::VectorTag>(vars));
  get<2>(get<test_databox_tags::VectorTag4>(out_vars)) *=
      get<2>(get<test_databox_tags::VectorTag>(vars));
  return out_vars;
}
}  // namespace

namespace test_databox_tags {
struct MultiplyScalarByTwo : db::ComputeItemTag {
  using variables_tags =
      tmpl::list<test_databox_tags::ScalarTag2, test_databox_tags::VectorTag2>;
  static constexpr db::DataBoxString label = "MultiplyScalarByTwo";
  static constexpr auto function = multiply_scalar_by_two;
  using argument_tags = tmpl::list<test_databox_tags::ScalarTag>;
};

struct MultiplyScalarByFour : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "MultiplyScalarByFour";
  static constexpr auto function = multiply_scalar_by_four;
  using argument_tags = tmpl::list<test_databox_tags::ScalarTag2>;
};

struct MultiplyScalarByThree : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "MultiplyScalarByThree";
  static constexpr auto function = multiply_scalar_by_three;
  using argument_tags = tmpl::list<test_databox_tags::MultiplyScalarByFour>;
};

struct DivideScalarByThree : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "DivideScalarByThree";
  static constexpr auto function = divide_scalar_by_three;
  using argument_tags = tmpl::list<test_databox_tags::MultiplyScalarByThree>;
};

struct DivideScalarByTwo : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "DivideScalarByTwo";
  static constexpr auto function = divide_scalar_by_two;
  using argument_tags = tmpl::list<test_databox_tags::DivideScalarByThree>;
};

struct MultiplyVariablesByTwo : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "MultiplyVariablesByTwo";
  static constexpr auto function = multiply_variables_by_two;
  using argument_tags = tmpl::list<Tags::Variables<
      tmpl::list<test_databox_tags::ScalarTag, test_databox_tags::VectorTag>>>;
};
}  // namespace test_databox_tags

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.Variables",
                  "[Unit][DataStructures]") {
  auto box = db::create<
      db::AddSimpleTags<Tags::Variables<tmpl::list<
          test_databox_tags::ScalarTag, test_databox_tags::VectorTag>>>,
      db::AddComputeTags<test_databox_tags::MultiplyScalarByTwo,
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

namespace {
struct Tag1 : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString label = "Tag1";
};
struct Tag2 : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString label = "Tag2";
};
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.Variables2",
                  "[Unit][DataStructures]") {
  auto box =
      db::create<db::AddSimpleTags<Tags::Variables<tmpl::list<Tag1, Tag2>>>>(
          Variables<tmpl::list<Tag1, Tag2>>(1, 1.));

  db::mutate<Tags::Variables<tmpl::list<Tag1, Tag2>>>(
      box, [](auto& vars) { vars = Variables<tmpl::list<Tag1, Tag2>>(1, 2.); });
  CHECK(db::get<Tag1>(box) == Scalar<DataVector>(DataVector(1, 2.)));
}

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.reset_compute_items",
                  "[Unit][DataStructures]") {
  auto box =
      db::create<db::AddSimpleTags<
                     test_databox_tags::Tag0, test_databox_tags::Tag1,
                     test_databox_tags::Tag2,
                     Tags::Variables<tmpl::list<test_databox_tags::ScalarTag,
                                                test_databox_tags::VectorTag>>>,
                 db::AddComputeTags<test_databox_tags::ComputeTag0,
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
      db::AddSimpleTags<
          test_databox_tags::Tag0,
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

namespace ExtraResetTags {
struct Var : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString label = "Var";
};
struct Int : db::DataBoxTag {
  using type = int;
  static constexpr db::DataBoxString label = "Int";
};
struct CheckReset : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "CheckReset";
  static auto function(
      const ::Variables<tmpl::list<Var>>& /*unused*/) noexcept {
    static bool first_call = true;
    CHECK(first_call);
    first_call = false;
    return 0;
  }
  using argument_tags = tmpl::list<Tags::Variables<tmpl::list<Var>>>;
};
}  // namespace ExtraResetTags

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.Variables.extra_reset",
                  "[Unit][DataStructures]") {
  auto box = db::create<
      db::AddSimpleTags<ExtraResetTags::Int,
                        Tags::Variables<tmpl::list<ExtraResetTags::Var>>>,
      db::AddComputeTags<ExtraResetTags::CheckReset>>(
      1, Variables<tmpl::list<ExtraResetTags::Var>>(2, 3.));
  CHECK(db::get<ExtraResetTags::CheckReset>(box) == 0);
  db::mutate<ExtraResetTags::Int>(box, [](int&){});
  CHECK(db::get<ExtraResetTags::CheckReset>(box) == 0);
}

namespace {
/// [mutate_apply_apply_struct_example]
struct test_databox_mutate_apply {
  static void apply(const gsl::not_null<Scalar<DataVector>*> scalar,
                    const gsl::not_null<tnsr::I<DataVector, 3>*> vector,
                    const std::string& tag2) {
    scalar->get() *= 2.0;
    get<0>(*vector) *= 3.0;
    get<1>(*vector) *= 4.0;
    get<2>(*vector) *= 5.0;
    CHECK(tag2 == "My Sample String"s);
  }
};
/// [mutate_apply_apply_struct_example]
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.mutate_apply",
                  "[Unit][DataStructures]") {
  auto box =
      db::create<db::AddSimpleTags<
                     test_databox_tags::Tag0, test_databox_tags::Tag1,
                     test_databox_tags::Tag2,
                     Tags::Variables<tmpl::list<test_databox_tags::ScalarTag,
                                                test_databox_tags::VectorTag>>>,
                 db::AddComputeTags<test_databox_tags::ComputeTag0,
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
        get<0>(*vector) *= 3.0;
        get<1>(*vector) *= 4.0;
        get<2>(*vector) *= 5.0;
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
        get<0>(get<test_databox_tags::VectorTag>(*vars)) *= 3.0;
        get<1>(get<test_databox_tags::VectorTag>(*vars)) *= 4.0;
        get<2>(get<test_databox_tags::VectorTag>(*vars)) *= 5.0;
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

namespace {
static_assert(
    cpp17::is_same_v<
        tmpl::list<test_databox_tags::ComputeTag0,
                   test_databox_tags::ComputeTag1,
                   test_databox_tags::MultiplyScalarByTwo>,
        db::get_compute_items<
            tmpl::list<test_databox_tags::Tag0, test_databox_tags::ComputeTag0,
                       test_databox_tags::Tag1, test_databox_tags::ComputeTag1,
                       test_databox_tags::MultiplyScalarByTwo>>>,
    "Failed testing db::get_compute_items");
static_assert(
    cpp17::is_same_v<
        tmpl::list<test_databox_tags::Tag0, test_databox_tags::Tag1>,
        db::get_items<
            tmpl::list<test_databox_tags::Tag0, test_databox_tags::ComputeTag0,
                       test_databox_tags::Tag1, test_databox_tags::ComputeTag1,
                       test_databox_tags::MultiplyScalarByTwo>>>,
    "Failed testing db::get_items");

static_assert(
    cpp17::is_same_v<
        db::compute_databox_type<tmpl::list<
            test_databox_tags::Tag0, test_databox_tags::Tag1,
            Tags::Variables<tmpl::list<test_databox_tags::ScalarTag,
                                       test_databox_tags::VectorTag>>,
            test_databox_tags::ComputeTag0,
            test_databox_tags::MultiplyScalarByTwo>>,
        db::DataBox<tmpl::list<
            test_databox_tags::Tag0, test_databox_tags::Tag1,
            Tags::Variables<brigand::list<test_databox_tags::ScalarTag,
                                          test_databox_tags::VectorTag>>,
            test_databox_tags::ScalarTag, test_databox_tags::VectorTag,
            test_databox_tags::ComputeTag0,
            test_databox_tags::MultiplyScalarByTwo,
            test_databox_tags::ScalarTag2, test_databox_tags::VectorTag2>>>,
    "Failed testing db::compute_databox_type");
}  // namespace

namespace {
void multiply_by_two_mutate(const gsl::not_null<std::vector<double>*> t,
                            const double value) {
  if (t->empty()) {
    t->resize(10);
  }
  for (auto& p : *t) {
    p = 2.0 * value;
  }
}
std::vector<double> multiply_by_two_non_mutate(const double value) {
  return std::vector<double>(10, 2.0 * value);
}

/// [databox_mutating_compute_item_function]
void mutate_variables(
    const gsl::not_null<Variables<tmpl::list<test_databox_tags::ScalarTag,
                                             test_databox_tags::VectorTag>>*>
        t,
    const double value) {
  if (t->number_of_grid_points() != 10) {
    *t = Variables<
        tmpl::list<test_databox_tags::ScalarTag, test_databox_tags::VectorTag>>(
        10, 0.0);
  }
  for (auto& p : get<test_databox_tags::ScalarTag>(*t)) {
    p = 2.0 * value;
  }
  for (auto& p : get<test_databox_tags::VectorTag>(*t)) {
    p = 3.0 * value;
  }
}
/// [databox_mutating_compute_item_function]
}  // namespace

namespace test_databox_tags {
struct MutateComputeTag0 : db::ComputeItemTag {
  using return_type = std::vector<double>;
  static constexpr db::DataBoxString label = "MutateComputeTag0";
  static constexpr auto function = multiply_by_two_mutate;
  using argument_tags = tmpl::list<Tag0>;
};
struct NonMutateComputeTag0 : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "NonMutateComputeTag0";
  static constexpr auto function = multiply_by_two_non_mutate;
  using argument_tags = tmpl::list<Tag0>;
};
/// [databox_mutating_compute_item_tag]
struct MutateVariablesCompute : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "MutateVariablesCompute";
  static constexpr auto function = mutate_variables;
  using return_type = Variables<
      tmpl::list<test_databox_tags::ScalarTag, test_databox_tags::VectorTag>>;
  using argument_tags = tmpl::list<Tag0>;
};
/// [databox_mutating_compute_item_tag]
}  // namespace test_databox_tags

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.mutating_compute_item",
                  "[Unit][DataStructures]") {
  auto original_box = db::create<
      db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                        test_databox_tags::Tag2>,
      db::AddComputeTags<test_databox_tags::MutateComputeTag0,
                         test_databox_tags::NonMutateComputeTag0,
                         test_databox_tags::MutateVariablesCompute,
                         test_databox_tags::ComputeTag0,
                         test_databox_tags::ComputeTag1>>(
      3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
  const double* const initial_data_location_mutating =
      db::get<test_databox_tags::MutateComputeTag0>(original_box).data();
  const double* const initial_data_location_non_mutating =
      db::get<test_databox_tags::NonMutateComputeTag0>(original_box).data();
  const std::array<const double* const, 4>
      initial_variables_data_location_mutate{
          {get<test_databox_tags::ScalarTag>(
               db::get<test_databox_tags::MutateVariablesCompute>(original_box))
               .get()
               .data(),
           get<0>(get<test_databox_tags::VectorTag>(
                      db::get<test_databox_tags::MutateVariablesCompute>(
                          original_box)))
               .data(),
           get<1>(get<test_databox_tags::VectorTag>(
                      db::get<test_databox_tags::MutateVariablesCompute>(
                          original_box)))
               .data(),
           get<2>(get<test_databox_tags::VectorTag>(
                      db::get<test_databox_tags::MutateVariablesCompute>(
                          original_box)))
               .data()}};

  CHECK(approx(db::get<test_databox_tags::ComputeTag0>(original_box)) ==
        3.14 * 2.0);
  CHECK_ITERABLE_APPROX(
      db::get<test_databox_tags::MutateComputeTag0>(original_box),
      std::vector<double>(10, 2.0 * 3.14));
  CHECK_ITERABLE_APPROX(
      db::get<test_databox_tags::NonMutateComputeTag0>(original_box),
      std::vector<double>(10, 2.0 * 3.14));
  CHECK_ITERABLE_APPROX(
      get<test_databox_tags::ScalarTag>(
          db::get<test_databox_tags::MutateVariablesCompute>(original_box)),
      Scalar<DataVector>(DataVector(10, 2.0 * 3.14)));
  CHECK_ITERABLE_APPROX(
      get<test_databox_tags::VectorTag>(
          db::get<test_databox_tags::MutateVariablesCompute>(original_box)),
      db::item_type<test_databox_tags::VectorTag>(DataVector(10, 3.0 * 3.14)));

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
  CHECK(approx(db::get<test_databox_tags::ComputeTag0>(original_box)) ==
        10.32 * 2.0);
  CHECK_ITERABLE_APPROX(
      db::get<test_databox_tags::MutateComputeTag0>(original_box),
      std::vector<double>(10, 2.0 * 10.32));
  CHECK(initial_data_location_mutating ==
        db::get<test_databox_tags::MutateComputeTag0>(original_box).data());
  CHECK_ITERABLE_APPROX(
      db::get<test_databox_tags::NonMutateComputeTag0>(original_box),
      std::vector<double>(10, 2.0 * 10.32));
  CHECK(initial_data_location_non_mutating !=
        db::get<test_databox_tags::MutateComputeTag0>(original_box).data());
  CHECK_ITERABLE_APPROX(
      get<test_databox_tags::ScalarTag>(
          db::get<test_databox_tags::MutateVariablesCompute>(original_box)),
      Scalar<DataVector>(DataVector(10, 2.0 * 10.32)));
  CHECK_ITERABLE_APPROX(
      get<test_databox_tags::VectorTag>(
          db::get<test_databox_tags::MutateVariablesCompute>(original_box)),
      db::item_type<test_databox_tags::VectorTag>(DataVector(10, 3.0 * 10.32)));

  // Check that the memory allocated by std::vector has not changed, which is
  // the key feature of mutating compute items.
  CHECK(
      initial_variables_data_location_mutate ==
      (std::array<const double* const, 4>{
          {get<test_databox_tags::ScalarTag>(
               db::get<test_databox_tags::MutateVariablesCompute>(original_box))
               .get()
               .data(),
           get<0>(get<test_databox_tags::VectorTag>(
                      db::get<test_databox_tags::MutateVariablesCompute>(
                          original_box)))
               .data(),
           get<1>(get<test_databox_tags::VectorTag>(
                      db::get<test_databox_tags::MutateVariablesCompute>(
                          original_box)))
               .data(),
           get<2>(get<test_databox_tags::VectorTag>(
                      db::get<test_databox_tags::MutateVariablesCompute>(
                          original_box)))
               .data()}}));
}

namespace DataBoxTest_detail {
struct vector : db::DataBoxTag {
  using type = tnsr::I<DataVector, 3, Frame::Grid>;
  static constexpr db::DataBoxString label = "vector";
};

struct scalar : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString label = "scalar";
};

struct vector2 : db::DataBoxTag {
  using type = tnsr::I<DataVector, 3, Frame::Grid>;
  static constexpr db::DataBoxString label = "vector2";
};
}  // namespace DataBoxTest_detail

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.data_on_slice.single",
                  "[Unit][DataStructures]") {
  const size_t x_extents = 2, y_extents = 3, z_extents = 4,
               vec_size = DataBoxTest_detail::vector::type::size();
  Index<3> extents(x_extents, y_extents, z_extents);
  auto box = db::create<db::AddSimpleTags<DataBoxTest_detail::vector>>([]() {
    Variables<tmpl::list<DataBoxTest_detail::vector>> vars(24, 0.);
    for (size_t s = 0; s < vars.size(); ++s) {
      // clang-tidy: do not use pointer arithmetic
      vars.data()[s] = s;  // NOLINT
    }
    return get<DataBoxTest_detail::vector>(vars);
  }());

  Variables<tmpl::list<DataBoxTest_detail::vector>> expected_vars_sliced_in_x(
      y_extents * z_extents, 0.),
      expected_vars_sliced_in_y(x_extents * z_extents, 0.),
      expected_vars_sliced_in_z(x_extents * y_extents, 0.);
  const size_t x_offset = 1, y_offset = 2, z_offset = 1;

  for (size_t s = 0; s < expected_vars_sliced_in_x.size(); ++s) {
    // clang-tidy: do not use pointer arithmetic
    expected_vars_sliced_in_x.data()[s] = x_offset + s * x_extents;  // NOLINT
  }
  for (size_t i = 0; i < vec_size; ++i) {
    for (size_t x = 0; x < x_extents; ++x) {
      for (size_t z = 0; z < z_extents; ++z) {
        // clang-tidy: do not use pointer arithmetic
        expected_vars_sliced_in_y
            .data()[x + x_extents * (z + z_extents * i)] =  // NOLINT
            i * extents.product() + x + x_extents * (y_offset + z * y_extents);
      }
    }
  }
  for (size_t i = 0; i < vec_size; ++i) {
    for (size_t x = 0; x < x_extents; ++x) {
      for (size_t y = 0; y < y_extents; ++y) {
        // clang-tidy: do not use pointer arithmetic
        expected_vars_sliced_in_z
            .data()[x + x_extents * (y + y_extents * i)] =  // NOLINT
            i * extents.product() + x + x_extents * (y + y_extents * z_offset);
      }
    }
  }
  CHECK(
      /// [data_on_slice]
      db::data_on_slice(box, extents, 0, x_offset,
                        tmpl::list<DataBoxTest_detail::vector>{})
      /// [data_on_slice]
      == expected_vars_sliced_in_x);
  CHECK(db::data_on_slice(box, extents, 1, y_offset,
                          tmpl::list<DataBoxTest_detail::vector>{}) ==
        expected_vars_sliced_in_y);
  CHECK(db::data_on_slice(box, extents, 2, z_offset,
                          tmpl::list<DataBoxTest_detail::vector>{}) ==
        expected_vars_sliced_in_z);
}

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.data_on_slice",
                  "[Unit][DataStructures]") {
  const size_t x_extents = 2, y_extents = 3, z_extents = 4,
               vec_size = DataBoxTest_detail::vector::type::size();
  Index<3> extents(x_extents, y_extents, z_extents);
  auto box = db::create<
      db::AddSimpleTags<DataBoxTest_detail::vector, DataBoxTest_detail::scalar,
                        DataBoxTest_detail::vector2>>(
      []() {
        Variables<tmpl::list<DataBoxTest_detail::vector>> vars(24, 0.);
        for (size_t s = 0; s < vars.size(); ++s) {
          // clang-tidy: do not use pointer arithmetic
          vars.data()[s] = s;  // NOLINT
        }
        return get<DataBoxTest_detail::vector>(vars);
      }(),
      Scalar<DataVector>(DataVector{8.9, 0.7, 6.7}),
      []() {
        Variables<tmpl::list<DataBoxTest_detail::vector>> vars(24, 0.);
        for (size_t s = 0; s < vars.size(); ++s) {
          // clang-tidy: do not use pointer arithmetic
          vars.data()[s] = s * 10.0;  // NOLINT
        }
        return get<DataBoxTest_detail::vector>(vars);
      }());

  Variables<tmpl::list<DataBoxTest_detail::vector>> expected_vars_sliced_in_x(
      y_extents * z_extents, 0.),
      expected_vars_sliced_in_y(x_extents * z_extents, 0.),
      expected_vars_sliced_in_z(x_extents * y_extents, 0.);
  const size_t x_offset = 1, y_offset = 2, z_offset = 1;

  for (size_t s = 0; s < expected_vars_sliced_in_x.size(); ++s) {
    // clang-tidy: do not use pointer arithmetic
    expected_vars_sliced_in_x.data()[s] = x_offset + s * x_extents;  // NOLINT
  }
  for (size_t i = 0; i < vec_size; ++i) {
    for (size_t x = 0; x < x_extents; ++x) {
      for (size_t z = 0; z < z_extents; ++z) {
        // clang-tidy: do not use pointer arithmetic
        expected_vars_sliced_in_y
            .data()[x + x_extents * (z + z_extents * i)] =  // NOLINT
            i * extents.product() + x + x_extents * (y_offset + z * y_extents);
      }
    }
  }
  for (size_t i = 0; i < vec_size; ++i) {
    for (size_t x = 0; x < x_extents; ++x) {
      for (size_t y = 0; y < y_extents; ++y) {
        // clang-tidy: do not use pointer arithmetic
        expected_vars_sliced_in_z
            .data()[x + x_extents * (y + y_extents * i)] =  // NOLINT
            i * extents.product() + x + x_extents * (y + y_extents * z_offset);
      }
    }
  }
  // x slice
  {
    const auto sliced0 = data_on_slice(
        box, extents, 0, x_offset,
        tmpl::list<DataBoxTest_detail::vector, DataBoxTest_detail::vector2>{});
    CHECK(get<DataBoxTest_detail::vector>(sliced0) ==
          get<DataBoxTest_detail::vector>(expected_vars_sliced_in_x));
    CHECK(get<DataBoxTest_detail::vector2>(sliced0) ==
          get<DataBoxTest_detail::vector>(
              Variables<tmpl::list<DataBoxTest_detail::vector>>(
                  expected_vars_sliced_in_x * 10.0)));
    const auto sliced1 = data_on_slice(
        box, extents, 0, x_offset, tmpl::list<DataBoxTest_detail::vector2>{});
    CHECK(get<DataBoxTest_detail::vector2>(sliced1) ==
          get<DataBoxTest_detail::vector>(
              Variables<tmpl::list<DataBoxTest_detail::vector>>(
                  expected_vars_sliced_in_x * 10.0)));
  }
  // y slice
  {
    const auto sliced0 = data_on_slice(
        box, extents, 1, y_offset,
        tmpl::list<DataBoxTest_detail::vector, DataBoxTest_detail::vector2>{});
    CHECK(get<DataBoxTest_detail::vector>(sliced0) ==
          get<DataBoxTest_detail::vector>(expected_vars_sliced_in_y));
    CHECK(get<DataBoxTest_detail::vector2>(sliced0) ==
          get<DataBoxTest_detail::vector>(
              Variables<tmpl::list<DataBoxTest_detail::vector>>(
                  expected_vars_sliced_in_y * 10.0)));
    const auto sliced1 = data_on_slice(
        box, extents, 1, y_offset, tmpl::list<DataBoxTest_detail::vector2>{});
    CHECK(get<DataBoxTest_detail::vector2>(sliced1) ==
          get<DataBoxTest_detail::vector>(
              Variables<tmpl::list<DataBoxTest_detail::vector>>(
                  expected_vars_sliced_in_y * 10.0)));
  }
  // z slice
  {
    const auto sliced0 = data_on_slice(
        box, extents, 2, z_offset,
        tmpl::list<DataBoxTest_detail::vector, DataBoxTest_detail::vector2>{});
    CHECK(get<DataBoxTest_detail::vector>(sliced0) ==
          get<DataBoxTest_detail::vector>(expected_vars_sliced_in_z));
    CHECK(get<DataBoxTest_detail::vector2>(sliced0) ==
          get<DataBoxTest_detail::vector>(
              Variables<tmpl::list<DataBoxTest_detail::vector>>(
                  expected_vars_sliced_in_z * 10.0)));
    const auto sliced1 = data_on_slice(
        box, extents, 2, z_offset, tmpl::list<DataBoxTest_detail::vector2>{});
    CHECK(get<DataBoxTest_detail::vector2>(sliced1) ==
          get<DataBoxTest_detail::vector>(
              Variables<tmpl::list<DataBoxTest_detail::vector>>(
                  expected_vars_sliced_in_z * 10.0)));
  }
}

namespace test_subitems {
// We can't use raw fundamental types as subitems because subitems
// need to have a reference-like nature.
template <typename T>
class Boxed {
 public:
  explicit Boxed(std::shared_ptr<T> data) noexcept : data_(std::move(data)) {}
  Boxed() = default;
  // The multiple copy constructors (assignment operators) are needed
  // to prevent users from modifying compute item values.
  Boxed(const Boxed&) = delete;
  Boxed(Boxed&) = default;
  Boxed(Boxed&&) = default;
  Boxed& operator=(const Boxed&) = delete;
  Boxed& operator=(Boxed&) = default;
  Boxed& operator=(Boxed&&) = default;
  ~Boxed() = default;

  T& operator*() noexcept { return *data_; }
  const T& operator*() const noexcept { return *data_; }

  // clang-tidy: no non-const references
  void pup(PUP::er& p) noexcept {  // NOLINT
    if (p.isUnpacking()) {
      T t{};
      p | t;
      data_ = std::make_shared<T>(std::move(t));
    } else {
      p | *data_;
    }
  }

 private:
  std::shared_ptr<T> data_;
};

template <size_t N, bool Compute = false, bool DependsOnComputeItem = false>
struct Parent : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Parent";
  using type = std::pair<Boxed<int>, Boxed<double>>;
};
template <size_t N, bool DependsOnComputeItem>
struct Parent<N, true, DependsOnComputeItem> : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "Parent";
  static auto function(
      const std::pair<Boxed<int>, Boxed<double>>& arg) noexcept {
    count++;
    return std::make_pair(
        Boxed<int>(std::make_shared<int>(*arg.first + 1)),
        Boxed<double>(std::make_shared<double>(*arg.second * 2.)));
  }
  using argument_tags = tmpl::list<Parent<N - 1, DependsOnComputeItem>>;
  static int count;
};

template <size_t N, bool DependsOnComputeItem>
int Parent<N, true, DependsOnComputeItem>::count = 0;

template <size_t N>
struct First : db::DataBoxTag {
  static constexpr db::DataBoxString label = "First";
  using type = Boxed<int>;

  static constexpr size_t index = 0;
};
template <size_t N>
struct Second : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Second";
  using type = Boxed<double>;

  static constexpr size_t index = 1;
};
}  // namespace test_subitems

namespace db {
template <size_t N, bool Compute, bool DependsOnComputeItem>
struct Subitems<test_subitems::Parent<N, Compute, DependsOnComputeItem>> {
  using type = tmpl::list<test_subitems::First<N>, test_subitems::Second<N>>;
  using tag = test_subitems::Parent<N, Compute>;

  template <typename Subtag>
  static void create_item(
      const gsl::not_null<item_type<tag>*> parent_value,
      const gsl::not_null<item_type<Subtag>*> sub_value) noexcept {
    *sub_value = std::get<Subtag::index>(*parent_value);
  }

  template <typename Subtag>
  static item_type<Subtag> create_compute_item(
      const item_type<tag>& parent_value) noexcept {
    // clang-tidy: do not use const_cast
    // We need a non-const object to set up the aliasing since in the
    // simple-item case the alias can be used to modify the original
    // item.  That should not be allowed for compute items, but the
    // DataBox will only allow access to a const version of the result
    // and we ensure in the definition of Boxed that that will not
    // allow modification of the original item.
    return const_cast<item_type<Subtag>&>(  // NOLINT
        std::get<Subtag::index>(parent_value));
  }
};
}  // namespace db

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.Subitems",
                  "[Unit][DataStructures]") {
  auto box = db::create<db::AddSimpleTags<test_subitems::Parent<0>>,
                        db::AddComputeTags<test_subitems::Parent<1, true>>>(
      std::make_pair(
          test_subitems::Boxed<int>(std::make_shared<int>(5)),
          test_subitems::Boxed<double>(std::make_shared<double>(3.5))));

  CHECK(*db::get<test_subitems::First<0>>(box) == 5);
  CHECK(*db::get<test_subitems::First<1>>(box) == 6);
  CHECK(*db::get<test_subitems::Second<0>>(box) == 3.5);
  CHECK(*db::get<test_subitems::Second<1>>(box) == 7);

  db::mutate<test_subitems::Second<0>>(
      box, [](test_subitems::Boxed<double>& x) noexcept { *x = 12.; });

  CHECK(*db::get<test_subitems::First<0>>(box) == 5);
  CHECK(*db::get<test_subitems::First<1>>(box) == 6);
  CHECK(*db::get<test_subitems::Second<0>>(box) == 12.);
  CHECK(*db::get<test_subitems::Second<1>>(box) == 24.);

  static_assert(
      cpp17::is_same_v<
          decltype(box),
          decltype(db::create_from<db::RemoveTags<test_subitems::Parent<2>>>(
              db::create_from<db::RemoveTags<>,
                              db::AddSimpleTags<test_subitems::Parent<2>>>(
                  box, std::make_pair(
                           test_subitems::Boxed<int>(std::make_shared<int>(5)),
                           test_subitems::Boxed<double>(
                               std::make_shared<double>(3.5))))))>,
      "Failed testing that adding and removing a simple subitem does "
      "not change the type of the DataBox");

  static_assert(
      cpp17::is_same_v<
          decltype(box),
          decltype(db::create_from<
                   db::RemoveTags<test_subitems::Parent<2, true, true>>>(
              db::create_from<
                  db::RemoveTags<>, db::AddSimpleTags<>,
                  db::AddComputeTags<test_subitems::Parent<2, true, true>>>(
                  box)))>,
      "Failed testing that adding and removing a compute subitem does "
      "not change the type of the DataBox");
}

namespace {
// Test serialization of a DataBox with non-Subitem simple items only.
void serialization_non_subitem_simple_items() noexcept {
  auto serialization_test_box = db::create<
      db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                        test_databox_tags::Tag2>>(
      3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
  const double* before_0 =
      &db::get<test_databox_tags::Tag0>(serialization_test_box);
  const std::vector<double>* before_1 =
      &db::get<test_databox_tags::Tag1>(serialization_test_box);
  const std::string* before_2 =
      &db::get<test_databox_tags::Tag2>(serialization_test_box);

  auto deserialized_serialization_test_box =
      serialize_and_deserialize(serialization_test_box);
  CHECK(db::get<test_databox_tags::Tag0>(serialization_test_box) == 3.14);
  CHECK(db::get<test_databox_tags::Tag0>(deserialized_serialization_test_box) ==
        3.14);
  CHECK(before_0 == &db::get<test_databox_tags::Tag0>(serialization_test_box));
  CHECK(before_0 !=
        &db::get<test_databox_tags::Tag0>(deserialized_serialization_test_box));
  CHECK(db::get<test_databox_tags::Tag1>(serialization_test_box) ==
        std::vector<double>{8.7, 93.2, 84.7});
  CHECK(db::get<test_databox_tags::Tag1>(deserialized_serialization_test_box) ==
        std::vector<double>{8.7, 93.2, 84.7});
  CHECK(before_1 == &db::get<test_databox_tags::Tag1>(serialization_test_box));
  CHECK(before_1 !=
        &db::get<test_databox_tags::Tag1>(deserialized_serialization_test_box));
  CHECK(db::get<test_databox_tags::Tag2>(serialization_test_box) ==
        "My Sample String"s);
  CHECK(db::get<test_databox_tags::Tag2>(deserialized_serialization_test_box) ==
        "My Sample String"s);
  CHECK(before_2 == &db::get<test_databox_tags::Tag2>(serialization_test_box));
  CHECK(before_2 !=
        &db::get<test_databox_tags::Tag2>(deserialized_serialization_test_box));
}

// Test serialization of a DataBox with Subitem and non-Subitem simple items.
void serialization_subitems_simple_items() noexcept {
  auto serialization_test_box = db::create<
      db::AddSimpleTags<test_databox_tags::Tag0, test_subitems::Parent<0>,
                        test_databox_tags::Tag1, test_databox_tags::Tag2,
                        test_subitems::Parent<1>>>(
      3.14,
      std::make_pair(
          test_subitems::Boxed<int>(std::make_shared<int>(5)),
          test_subitems::Boxed<double>(std::make_shared<double>(3.5))),
      std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s,
      std::make_pair(
          test_subitems::Boxed<int>(std::make_shared<int>(9)),
          test_subitems::Boxed<double>(std::make_shared<double>(-4.5))));
  const double* before_0 =
      &db::get<test_databox_tags::Tag0>(serialization_test_box);
  const std::vector<double>* before_1 =
      &db::get<test_databox_tags::Tag1>(serialization_test_box);
  const std::string* before_2 =
      &db::get<test_databox_tags::Tag2>(serialization_test_box);
  const std::pair<test_subitems::Boxed<int>, test_subitems::Boxed<double>>*
      before_parent0 =
          &db::get<test_subitems::Parent<0>>(serialization_test_box);
  const test_subitems::Boxed<int>* before_parent0f =
      &db::get<test_subitems::First<0>>(serialization_test_box);
  const test_subitems::Boxed<double>* before_parent0s =
      &db::get<test_subitems::Second<0>>(serialization_test_box);
  const std::pair<test_subitems::Boxed<int>, test_subitems::Boxed<double>>*
      before_parent1 =
          &db::get<test_subitems::Parent<1>>(serialization_test_box);
  const test_subitems::Boxed<int>* before_parent1f =
      &db::get<test_subitems::First<1>>(serialization_test_box);
  const test_subitems::Boxed<double>* before_parent1s =
      &db::get<test_subitems::Second<1>>(serialization_test_box);

  auto deserialized_serialization_test_box =
      serialize_and_deserialize(serialization_test_box);
  CHECK(db::get<test_databox_tags::Tag0>(serialization_test_box) == 3.14);
  CHECK(db::get<test_databox_tags::Tag0>(deserialized_serialization_test_box) ==
        3.14);
  CHECK(before_0 == &db::get<test_databox_tags::Tag0>(serialization_test_box));
  CHECK(before_0 !=
        &db::get<test_databox_tags::Tag0>(deserialized_serialization_test_box));
  CHECK(*db::get<test_subitems::First<0>>(serialization_test_box) == 5);
  CHECK(*db::get<test_subitems::Second<0>>(serialization_test_box) == 3.5);
  CHECK(*db::get<test_subitems::First<0>>(
            deserialized_serialization_test_box) == 5);
  CHECK(*db::get<test_subitems::Second<0>>(
            deserialized_serialization_test_box) == 3.5);
  CHECK(before_parent0 ==
        &db::get<test_subitems::Parent<0>>(serialization_test_box));
  CHECK(before_parent0 != &db::get<test_subitems::Parent<0>>(
                              deserialized_serialization_test_box));
  CHECK(before_parent0f ==
        &db::get<test_subitems::First<0>>(serialization_test_box));
  CHECK(before_parent0f !=
        &db::get<test_subitems::First<0>>(deserialized_serialization_test_box));
  CHECK(before_parent0s ==
        &db::get<test_subitems::Second<0>>(serialization_test_box));
  CHECK(before_parent0s != &db::get<test_subitems::Second<0>>(
                               deserialized_serialization_test_box));
  CHECK(db::get<test_databox_tags::Tag1>(serialization_test_box) ==
        std::vector<double>{8.7, 93.2, 84.7});
  CHECK(db::get<test_databox_tags::Tag1>(deserialized_serialization_test_box) ==
        std::vector<double>{8.7, 93.2, 84.7});
  CHECK(before_1 == &db::get<test_databox_tags::Tag1>(serialization_test_box));
  CHECK(before_1 !=
        &db::get<test_databox_tags::Tag1>(deserialized_serialization_test_box));
  CHECK(db::get<test_databox_tags::Tag2>(serialization_test_box) ==
        "My Sample String"s);
  CHECK(db::get<test_databox_tags::Tag2>(deserialized_serialization_test_box) ==
        "My Sample String"s);
  CHECK(before_2 == &db::get<test_databox_tags::Tag2>(serialization_test_box));
  CHECK(before_2 !=
        &db::get<test_databox_tags::Tag2>(deserialized_serialization_test_box));
  CHECK(*db::get<test_subitems::First<1>>(serialization_test_box) == 9);
  CHECK(*db::get<test_subitems::Second<1>>(serialization_test_box) == -4.5);
  CHECK(*db::get<test_subitems::First<1>>(
            deserialized_serialization_test_box) == 9);
  CHECK(*db::get<test_subitems::Second<1>>(
            deserialized_serialization_test_box) == -4.5);
  CHECK(before_parent1 ==
        &db::get<test_subitems::Parent<1>>(serialization_test_box));
  CHECK(before_parent1 != &db::get<test_subitems::Parent<1>>(
                              deserialized_serialization_test_box));
  CHECK(before_parent1f ==
        &db::get<test_subitems::First<1>>(serialization_test_box));
  CHECK(before_parent1f !=
        &db::get<test_subitems::First<1>>(deserialized_serialization_test_box));
  CHECK(before_parent1s ==
        &db::get<test_subitems::Second<1>>(serialization_test_box));
  CHECK(before_parent1s != &db::get<test_subitems::Second<1>>(
                               deserialized_serialization_test_box));
}

template <int Id>
struct CountingFunc {
  static double apply() {
    count++;
    return 8.2;
  }
  static int count;
};

template <int Id>
int CountingFunc<Id>::count = 0;

template <int Id>
struct CountingTag : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "CountingTag";
  static constexpr auto function = CountingFunc<Id>::apply;
  using argument_tags = tmpl::list<>;
};

template <size_t SecondId>
struct CountingTagDouble : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "CountingTag";
  static double function(const test_subitems::Boxed<double>& t) {
    count++;
    return *t * 6.0;
  }
  using argument_tags = tmpl::list<test_subitems::Second<SecondId>>;
  static int count;
};

template <size_t SecondId>
int CountingTagDouble<SecondId>::count = 0;

// Test serialization of a DataBox with Subitem compute items, one is
// evaluated before serialization, one is after.
// clang-tidy: this function is too long. Yes, well we need to check lots
void serialization_subitem_compute_items() noexcept {  // NOLINT
  auto serialization_test_box = db::create<
      db::AddSimpleTags<test_databox_tags::Tag0, test_subitems::Parent<0>,
                        test_databox_tags::Tag1, test_databox_tags::Tag2,
                        test_subitems::Parent<1>>,
      db::AddComputeTags<CountingTag<1>, test_databox_tags::ComputeTag0,
                         test_subitems::Parent<2, true>,
                         test_databox_tags::ComputeTag1,
                         test_subitems::Parent<3, true, true>, CountingTag<0>,
                         CountingTagDouble<2>, CountingTagDouble<3>>>(
      3.14,
      std::make_pair(
          test_subitems::Boxed<int>(std::make_shared<int>(5)),
          test_subitems::Boxed<double>(std::make_shared<double>(3.5))),
      std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s,
      std::make_pair(
          test_subitems::Boxed<int>(std::make_shared<int>(9)),
          test_subitems::Boxed<double>(std::make_shared<double>(-4.5))));
  const double* before_0 =
      &db::get<test_databox_tags::Tag0>(serialization_test_box);
  const std::vector<double>* before_1 =
      &db::get<test_databox_tags::Tag1>(serialization_test_box);
  const std::string* before_2 =
      &db::get<test_databox_tags::Tag2>(serialization_test_box);
  const std::pair<test_subitems::Boxed<int>, test_subitems::Boxed<double>>*
      before_parent0 =
          &db::get<test_subitems::Parent<0>>(serialization_test_box);
  const test_subitems::Boxed<int>* before_parent0f =
      &db::get<test_subitems::First<0>>(serialization_test_box);
  const test_subitems::Boxed<double>* before_parent0s =
      &db::get<test_subitems::Second<0>>(serialization_test_box);
  const std::pair<test_subitems::Boxed<int>, test_subitems::Boxed<double>>*
      before_parent1 =
          &db::get<test_subitems::Parent<1>>(serialization_test_box);
  const test_subitems::Boxed<int>* before_parent1f =
      &db::get<test_subitems::First<1>>(serialization_test_box);
  const test_subitems::Boxed<double>* before_parent1s =
      &db::get<test_subitems::Second<1>>(serialization_test_box);
  CHECK(db::get<test_databox_tags::ComputeTag0>(serialization_test_box) ==
        6.28);
  const double* before_compute_tag0 =
      &db::get<test_databox_tags::ComputeTag0>(serialization_test_box);
  CHECK(CountingFunc<0>::count == 0);
  CHECK(CountingFunc<1>::count == 0);
  CHECK(db::get<CountingTag<0>>(serialization_test_box) == 8.2);
  const double* before_counting_tag0 =
      &db::get<CountingTag<0>>(serialization_test_box);
  CHECK(CountingFunc<0>::count == 1);
  CHECK(CountingFunc<1>::count == 0);

  CHECK(test_subitems::Parent<2, true>::count == 0);
  CHECK(test_subitems::Parent<3, true, true>::count == 0);
  const std::pair<test_subitems::Boxed<int>, test_subitems::Boxed<double>>*
      before_parent2 =
          &db::get<test_subitems::Parent<2, true>>(serialization_test_box);
  CHECK(test_subitems::Parent<2, true>::count == 1);
  CHECK(test_subitems::Parent<3, true, true>::count == 0);
  const test_subitems::Boxed<int>* before_parent2_first =
      &db::get<test_subitems::First<2>>(serialization_test_box);
  const test_subitems::Boxed<double>* before_parent2_second =
      &db::get<test_subitems::Second<2>>(serialization_test_box);

  // Check we are correctly pointing into parent
  CHECK(&*(db::get<test_subitems::Parent<2, true>>(serialization_test_box)
               .first) ==
        &*db::get<test_subitems::First<2>>(serialization_test_box));
  CHECK(&*(db::get<test_subitems::Parent<2, true>>(serialization_test_box)
               .second) ==
        &*db::get<test_subitems::Second<2>>(serialization_test_box));

  CHECK(*(db::get<test_subitems::Parent<2, true>>(serialization_test_box)
              .first) == 10);
  CHECK(*(db::get<test_subitems::Parent<2, true>>(serialization_test_box)
              .second) == -9.0);

  CHECK(*db::get<test_subitems::First<2>>(serialization_test_box) == 10);
  CHECK(*db::get<test_subitems::Second<2>>(serialization_test_box) == -9.0);
  CHECK(test_subitems::Parent<2, true>::count == 1);
  CHECK(test_subitems::Parent<3, true, true>::count == 0);

  // Check compute items that take subitems
  CHECK(CountingTagDouble<2>::count == 0);
  CHECK(db::get<CountingTagDouble<2>>(serialization_test_box) == -9.0 * 6.0);
  CHECK(CountingTagDouble<2>::count == 1);
  const double* const before_compute_tag2 =
      &db::get<CountingTagDouble<2>>(serialization_test_box);

  auto deserialized_serialization_test_box =
      serialize_and_deserialize(serialization_test_box);
  CHECK(CountingFunc<0>::count == 1);
  CHECK(CountingFunc<1>::count == 0);
  CHECK(db::get<test_databox_tags::Tag0>(serialization_test_box) == 3.14);
  CHECK(db::get<test_databox_tags::Tag0>(deserialized_serialization_test_box) ==
        3.14);
  CHECK(before_0 == &db::get<test_databox_tags::Tag0>(serialization_test_box));
  CHECK(before_0 !=
        &db::get<test_databox_tags::Tag0>(deserialized_serialization_test_box));
  CHECK(*db::get<test_subitems::First<0>>(serialization_test_box) == 5);
  CHECK(*db::get<test_subitems::Second<0>>(serialization_test_box) == 3.5);
  CHECK(*db::get<test_subitems::First<0>>(
            deserialized_serialization_test_box) == 5);
  CHECK(*db::get<test_subitems::Second<0>>(
            deserialized_serialization_test_box) == 3.5);
  CHECK(before_parent0 ==
        &db::get<test_subitems::Parent<0>>(serialization_test_box));
  CHECK(before_parent0 != &db::get<test_subitems::Parent<0>>(
                              deserialized_serialization_test_box));
  CHECK(before_parent0f ==
        &db::get<test_subitems::First<0>>(serialization_test_box));
  CHECK(before_parent0f !=
        &db::get<test_subitems::First<0>>(deserialized_serialization_test_box));
  CHECK(before_parent0s ==
        &db::get<test_subitems::Second<0>>(serialization_test_box));
  CHECK(before_parent0s != &db::get<test_subitems::Second<0>>(
                               deserialized_serialization_test_box));
  CHECK(db::get<test_databox_tags::Tag1>(serialization_test_box) ==
        std::vector<double>{8.7, 93.2, 84.7});
  CHECK(db::get<test_databox_tags::Tag1>(deserialized_serialization_test_box) ==
        std::vector<double>{8.7, 93.2, 84.7});
  CHECK(before_1 == &db::get<test_databox_tags::Tag1>(serialization_test_box));
  CHECK(before_1 !=
        &db::get<test_databox_tags::Tag1>(deserialized_serialization_test_box));
  CHECK(db::get<test_databox_tags::Tag2>(serialization_test_box) ==
        "My Sample String"s);
  CHECK(db::get<test_databox_tags::Tag2>(deserialized_serialization_test_box) ==
        "My Sample String"s);
  CHECK(before_2 == &db::get<test_databox_tags::Tag2>(serialization_test_box));
  CHECK(before_2 !=
        &db::get<test_databox_tags::Tag2>(deserialized_serialization_test_box));
  CHECK(*db::get<test_subitems::First<1>>(serialization_test_box) == 9);
  CHECK(*db::get<test_subitems::Second<1>>(serialization_test_box) == -4.5);
  CHECK(*db::get<test_subitems::First<1>>(
            deserialized_serialization_test_box) == 9);
  CHECK(*db::get<test_subitems::Second<1>>(
            deserialized_serialization_test_box) == -4.5);
  CHECK(before_parent1 ==
        &db::get<test_subitems::Parent<1>>(serialization_test_box));
  CHECK(before_parent1 != &db::get<test_subitems::Parent<1>>(
                              deserialized_serialization_test_box));
  CHECK(before_parent1f ==
        &db::get<test_subitems::First<1>>(serialization_test_box));
  CHECK(before_parent1f !=
        &db::get<test_subitems::First<1>>(deserialized_serialization_test_box));
  CHECK(before_parent1s ==
        &db::get<test_subitems::Second<1>>(serialization_test_box));
  CHECK(before_parent1s != &db::get<test_subitems::Second<1>>(
                               deserialized_serialization_test_box));
  // Check compute items
  CHECK(db::get<test_databox_tags::ComputeTag0>(
            deserialized_serialization_test_box) == 6.28);
  CHECK(&db::get<test_databox_tags::ComputeTag0>(
            deserialized_serialization_test_box) != before_compute_tag0);
  CHECK(db::get<test_databox_tags::ComputeTag1>(
            deserialized_serialization_test_box) == "My Sample String6.28"s);

  CHECK(CountingFunc<0>::count == 1);
  CHECK(CountingFunc<1>::count == 0);
  CHECK(db::get<CountingTag<0>>(serialization_test_box) == 8.2);
  CHECK(CountingFunc<0>::count == 1);
  CHECK(CountingFunc<1>::count == 0);
  CHECK(&db::get<CountingTag<0>>(serialization_test_box) ==
        before_counting_tag0);

  CHECK(db::get<CountingTag<0>>(deserialized_serialization_test_box) == 8.2);
  CHECK(CountingFunc<0>::count == 1);
  CHECK(CountingFunc<1>::count == 0);
  CHECK(&db::get<CountingTag<0>>(deserialized_serialization_test_box) !=
        before_counting_tag0);
  CHECK(CountingFunc<0>::count == 1);
  CHECK(CountingFunc<1>::count == 0);
  CHECK(db::get<CountingTag<1>>(deserialized_serialization_test_box) == 8.2);
  CHECK(CountingFunc<0>::count == 1);
  CHECK(CountingFunc<1>::count == 1);
  CHECK(db::get<CountingTag<1>>(serialization_test_box) == 8.2);
  CHECK(CountingFunc<0>::count == 1);
  CHECK(CountingFunc<1>::count == 2);
  CHECK(&db::get<CountingTag<1>>(serialization_test_box) !=
        &db::get<CountingTag<1>>(deserialized_serialization_test_box));

  CHECK(test_subitems::Parent<2, true>::count == 1);
  CHECK(test_subitems::Parent<3, true, true>::count == 0);
  CHECK(&db::get<test_subitems::Parent<2, true>>(serialization_test_box) ==
        before_parent2);
  // Check we are correctly pointing into parent
  CHECK(&*(db::get<test_subitems::Parent<2, true>>(serialization_test_box)
               .first) ==
        &*db::get<test_subitems::First<2>>(serialization_test_box));
  CHECK(&*(db::get<test_subitems::Parent<2, true>>(serialization_test_box)
               .second) ==
        &*db::get<test_subitems::Second<2>>(serialization_test_box));
  // Check that we did not reset the subitems items in the initial DataBox
  CHECK(&db::get<test_subitems::First<2>>(serialization_test_box) ==
        before_parent2_first);
  CHECK(&db::get<test_subitems::Second<2>>(serialization_test_box) ==
        before_parent2_second);
  CHECK(*(db::get<test_subitems::Parent<2, true>>(serialization_test_box)
              .first) == 10);
  CHECK(*(db::get<test_subitems::Parent<2, true>>(serialization_test_box)
              .second) == -9.0);
  CHECK(*(db::get<test_subitems::Parent<2, true>>(
              deserialized_serialization_test_box)
              .first) == 10);
  CHECK(&db::get<test_subitems::Parent<2, true>>(
            deserialized_serialization_test_box) != before_parent2);
  CHECK(*(db::get<test_subitems::Parent<2, true>>(
              deserialized_serialization_test_box)
              .second) == -9.0);
  CHECK(*db::get<test_subitems::First<2>>(
            deserialized_serialization_test_box) == 10);
  CHECK(*db::get<test_subitems::Second<2>>(
            deserialized_serialization_test_box) == -9.0);
  CHECK(test_subitems::Parent<2, true>::count == 1);
  CHECK(test_subitems::Parent<3, true, true>::count == 0);
  CHECK(&db::get<test_subitems::Parent<2, true>>(
            deserialized_serialization_test_box) != before_parent2);
  // Check pointers in deserialized box
  CHECK(&db::get<test_subitems::First<2>>(
            deserialized_serialization_test_box) != before_parent2_first);
  CHECK(&db::get<test_subitems::Second<2>>(
            deserialized_serialization_test_box) != before_parent2_second);
  // Check we are correctly pointing into new parent and not old
  CHECK(
      &*(db::get<test_subitems::Parent<2, true>>(
             deserialized_serialization_test_box)
             .first) ==
      &*db::get<test_subitems::First<2>>(deserialized_serialization_test_box));
  CHECK(
      &*(db::get<test_subitems::Parent<2, true>>(
             deserialized_serialization_test_box)
             .second) ==
      &*db::get<test_subitems::Second<2>>(deserialized_serialization_test_box));
  CHECK(&*(db::get<test_subitems::Parent<2, true>>(
               deserialized_serialization_test_box)
               .first) !=
        &*db::get<test_subitems::First<2>>(serialization_test_box));
  CHECK(&*(db::get<test_subitems::Parent<2, true>>(
               deserialized_serialization_test_box)
               .second) !=
        &*db::get<test_subitems::Second<2>>(serialization_test_box));

  CHECK(*(db::get<test_subitems::Parent<3, true, true>>(serialization_test_box)
              .first) == 11);
  CHECK(test_subitems::Parent<2, true>::count == 1);
  CHECK(test_subitems::Parent<3, true, true>::count == 1);
  CHECK(*(db::get<test_subitems::Parent<3, true, true>>(serialization_test_box)
              .second) == -18.0);
  CHECK(test_subitems::Parent<2, true>::count == 1);
  CHECK(test_subitems::Parent<3, true, true>::count == 1);
  CHECK(*db::get<test_subitems::First<3>>(serialization_test_box) == 11);
  CHECK(*db::get<test_subitems::Second<3>>(serialization_test_box) == -18.0);
  CHECK(test_subitems::Parent<2, true>::count == 1);
  CHECK(test_subitems::Parent<3, true, true>::count == 1);
  CHECK(*(db::get<test_subitems::Parent<3, true, true>>(
              deserialized_serialization_test_box)
              .first) == 11);
  CHECK(test_subitems::Parent<2, true>::count == 1);
  CHECK(test_subitems::Parent<3, true, true>::count == 2);
  CHECK(*(db::get<test_subitems::Parent<3, true, true>>(
              deserialized_serialization_test_box)
              .second) == -18.0);
  CHECK(*db::get<test_subitems::First<3>>(
            deserialized_serialization_test_box) == 11);
  CHECK(*db::get<test_subitems::Second<3>>(
            deserialized_serialization_test_box) == -18.0);
  CHECK(test_subitems::Parent<2, true>::count == 1);
  CHECK(test_subitems::Parent<3, true, true>::count == 2);

  // Check that all the Parent<3> related objects point to the right place
  CHECK(
      &*(db::get<test_subitems::Parent<3, true, true>>(
             deserialized_serialization_test_box)
             .first) ==
      &*db::get<test_subitems::First<3>>(deserialized_serialization_test_box));
  CHECK(
      &*(db::get<test_subitems::Parent<3, true, true>>(
             deserialized_serialization_test_box)
             .second) ==
      &*db::get<test_subitems::Second<3>>(deserialized_serialization_test_box));
  CHECK(&*(db::get<test_subitems::Parent<3, true, true>>(serialization_test_box)
               .first) ==
        &*db::get<test_subitems::First<3>>(serialization_test_box));
  CHECK(&*(db::get<test_subitems::Parent<3, true, true>>(serialization_test_box)
               .second) ==
        &*db::get<test_subitems::Second<3>>(serialization_test_box));
  CHECK(
      &*db::get<test_subitems::First<3>>(deserialized_serialization_test_box) !=
      &*db::get<test_subitems::First<3>>(serialization_test_box));
  CHECK(&*db::get<test_subitems::Second<3>>(
            deserialized_serialization_test_box) !=
        &*db::get<test_subitems::Second<3>>(serialization_test_box));

  // Check compute items that depend on the subitems
  CHECK(CountingTagDouble<2>::count == 1);
  CHECK(db::get<CountingTagDouble<2>>(serialization_test_box) == -9.0 * 6.0);
  CHECK(before_compute_tag2 ==
        &db::get<CountingTagDouble<2>>(serialization_test_box));
  CHECK(db::get<CountingTagDouble<2>>(deserialized_serialization_test_box) ==
        -9.0 * 6.0);
  CHECK(before_compute_tag2 !=
        &db::get<CountingTagDouble<2>>(deserialized_serialization_test_box));
  CHECK(CountingTagDouble<2>::count == 1);

  CHECK(CountingTagDouble<3>::count == 0);
  CHECK(db::get<CountingTagDouble<3>>(serialization_test_box) == -18.0 * 6.0);
  CHECK(db::get<CountingTagDouble<3>>(deserialized_serialization_test_box) ==
        -18.0 * 6.0);
  CHECK(&db::get<CountingTagDouble<3>>(serialization_test_box) !=
        &db::get<CountingTagDouble<3>>(deserialized_serialization_test_box));
  CHECK(CountingTagDouble<3>::count == 2);

  // Mutate subitems 1 in deserialized to see that changes propagate correctly
  db::mutate<test_subitems::Second<1>>(
      serialization_test_box, [](test_subitems::Boxed<double> & x) noexcept {
        *x = 12.;
      });
  CHECK(test_subitems::Parent<2, true>::count == 1);
  CHECK(CountingTagDouble<2>::count == 1);
  CHECK(db::get<CountingTagDouble<2>>(serialization_test_box) == 24.0 * 6.0);
  CHECK(test_subitems::Parent<2, true>::count == 2);
  CHECK(CountingTagDouble<2>::count == 2);
  CHECK(CountingTagDouble<3>::count == 2);
  CHECK(db::get<CountingTagDouble<3>>(serialization_test_box) == 48.0 * 6.0);
  CHECK(CountingTagDouble<3>::count == 3);

  db::mutate<test_subitems::Second<1>>(
      deserialized_serialization_test_box, [](test_subitems::Boxed<double> &
                                              x) noexcept { *x = -7.; });
  CHECK(test_subitems::Parent<2, true>::count == 2);
  CHECK(CountingTagDouble<2>::count == 2);
  CHECK(db::get<CountingTagDouble<2>>(deserialized_serialization_test_box) ==
        -14.0 * 6.0);
  CHECK(test_subitems::Parent<2, true>::count == 3);
  CHECK(CountingTagDouble<2>::count == 3);
  CHECK(CountingTagDouble<3>::count == 3);
  CHECK(db::get<CountingTagDouble<3>>(deserialized_serialization_test_box) ==
        -28.0 * 6.0);
  CHECK(CountingTagDouble<3>::count == 4);

  // Check things didn't get modified in the original DataBox
  CHECK(test_subitems::Parent<2, true>::count == 3);
  CHECK(CountingTagDouble<2>::count == 3);
  CHECK(db::get<CountingTagDouble<2>>(serialization_test_box) == 24.0 * 6.0);
  CHECK(test_subitems::Parent<2, true>::count == 3);
  CHECK(CountingTagDouble<2>::count == 3);
  CHECK(CountingTagDouble<3>::count == 4);
  CHECK(db::get<CountingTagDouble<3>>(serialization_test_box) == 48.0 * 6.0);
  CHECK(CountingTagDouble<3>::count == 4);

  CountingFunc<0>::count = 0;
  CountingFunc<1>::count = 0;
  CountingTagDouble<2>::count = 0;
  CountingTagDouble<3>::count = 0;
  test_subitems::Parent<2, true>::count = 0;
  test_subitems::Parent<3, true, true>::count = 0;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.Serialization",
                  "[Unit][DataStructures]") {
  serialization_non_subitem_simple_items();
  serialization_subitems_simple_items();
  serialization_subitem_compute_items();
}
