// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <pup.h>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/DataOnSlice.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/SubitemTag.hpp"
#include "DataStructures/DataBox/Subitems.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
struct NoSuchType;
template <typename TagsList>
class Variables;
template <typename X, typename Symm, typename IndexList>
class Tensor;

namespace {
void multiply_by_two(const gsl::not_null<double*> result, const double value) {
  *result = 2.0 * value;
}

void append_word(const gsl::not_null<std::string*> result,
                 const std::string& text, const double value) {
  std::stringstream ss;
  ss << value;
  *result = text + ss.str();
}

namespace test_databox_tags {
// [databox_tag_example]
struct Tag0 : db::SimpleTag {
  using type = double;
};
// [databox_tag_example]
struct Tag1 : db::SimpleTag {
  using type = std::vector<double>;
};
struct Tag2Base : db::BaseTag {};
struct Tag2 : db::SimpleTag, Tag2Base {
  using type = std::string;
};
struct Tag3 : db::SimpleTag {
  using type = std::string;
};

struct Tag4 : db::SimpleTag {
  using type = double;
};

struct Tag4Compute : Tag4, db::ComputeTag {
  using base = Tag4;
  using return_type = double;
  static constexpr auto function = multiply_by_two;
  using argument_tags = tmpl::list<Tag0>;
};

struct Tag5 : db::SimpleTag {
  using type = std::string;
};

struct Tag5Compute : Tag5, db::ComputeTag {
  using base = Tag5;
  using return_type = std::string;
  static constexpr auto function = append_word;
  using argument_tags = tmpl::list<Tag2, Tag4>;
};

struct Tag6 : db::SimpleTag {
  using type = std::string;
};

struct Tag6Compute : Tag6, db::ComputeTag {
  using base = Tag6;
  using return_type = std::string;
  static void function(gsl::not_null<std::string*> result,
                       const std::string& s) noexcept {
    *result = s;
  }
  using argument_tags = tmpl::list<Tag2Base>;
};

// [compute_item_tag_function]
struct Lambda0 : db::SimpleTag {
  using type = double;
};

struct Lambda0Compute : Lambda0, db::ComputeTag {
  using base = Lambda0;
  using return_type = double;
  static constexpr void function(const gsl::not_null<double*> result,
                                 const double a) {
    *result = 3.0 * a;
  }
  using argument_tags = tmpl::list<Tag0>;
};
// [compute_item_tag_function]

struct Lambda1 : db::SimpleTag {
  using type = double;
};

// [compute_item_tag_no_tags]
struct Lambda1Compute : Lambda1, db::ComputeTag {
  using base = Lambda1;
  using return_type = double;
  static constexpr void function(const gsl::not_null<double*> result) {
    *result = 7.0;
  }
  using argument_tags = tmpl::list<>;
};
// [compute_item_tag_no_tags]

// [databox_prefix_tag_example]
template <typename Tag>
struct TagPrefix : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
  static std::string name() noexcept {
    return "TagPrefix(" + db::tag_name<Tag>() + ")";
  }
};
// [databox_prefix_tag_example]

struct PointerBase : db::BaseTag {};

struct Pointer : PointerBase, db::SimpleTag {
  using type = std::unique_ptr<int>;
};

struct PointerToCounterBase : db::BaseTag {};

struct PointerToCounter : PointerToCounterBase, db::SimpleTag {
  using type = std::unique_ptr<int>;
};

struct PointerToCounterCompute : PointerToCounter, db::ComputeTag {
  using base = PointerToCounter;
  using return_type = std::unique_ptr<int>;
  static void function(const gsl::not_null<return_type*> result,
                       const int& p) noexcept {
    *result = std::make_unique<int>(p + 1);
  }
  using argument_tags = tmpl::list<Pointer>;
};

struct PointerToSum : db::SimpleTag {
  using type = std::unique_ptr<int>;
};

struct PointerToSumCompute : PointerToSum, db::ComputeTag {
  using base = PointerToSum;
  using return_type = std::unique_ptr<int>;
  static void function(const gsl::not_null<return_type*> ret, const int& arg,
                       const int& same_arg) noexcept {
    *ret = std::make_unique<int>(arg + same_arg);
  }
  using argument_tags = tmpl::list<PointerToCounter, PointerToCounterBase>;
};
}  // namespace test_databox_tags

using EmptyBox = decltype(db::create<db::AddSimpleTags<>>());
static_assert(std::is_same_v<decltype(db::create_from<db::RemoveTags<>>(
                                 std::declval<EmptyBox>())),
                             EmptyBox>,
              "Wrong create_from result type");

using Box_t = db::DataBox<tmpl::list<
    test_databox_tags::Tag0, test_databox_tags::Tag1, test_databox_tags::Tag2,
    test_databox_tags::TagPrefix<test_databox_tags::Tag0>,
    test_databox_tags::Tag4Compute, test_databox_tags::Tag5Compute>>;

static_assert(
    std::is_same<
        decltype(
            db::create_from<db::RemoveTags<test_databox_tags::Tag1>>(Box_t{})),
        db::DataBox<
            tmpl::list<test_databox_tags::Tag0, test_databox_tags::Tag2,
                       test_databox_tags::TagPrefix<test_databox_tags::Tag0>,
                       test_databox_tags::Tag4Compute,
                       test_databox_tags::Tag5Compute>>>::value,
    "Failed testing removal of item");

static_assert(
    std::is_same<
        decltype(
            db::create_from<db::RemoveTags<test_databox_tags::Tag5>>(Box_t{})),
        db::DataBox<
            tmpl::list<test_databox_tags::Tag0, test_databox_tags::Tag1,
                       test_databox_tags::Tag2,
                       test_databox_tags::TagPrefix<test_databox_tags::Tag0>,
                       test_databox_tags::Tag4Compute>>>::value,
    "Failed testing removal of compute item");

static_assert(std::is_same<decltype(db::create_from<db::RemoveTags<>>(Box_t{})),
                           Box_t>::value,
              "Failed testing no-op create_from");

void test_databox() noexcept {
  INFO("test databox");
  const auto create_original_box = []() {
    // [create_databox]
    auto original_box = db::create<
        db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                          test_databox_tags::Tag2>,
        db::AddComputeTags<test_databox_tags::Tag4Compute,
                           test_databox_tags::Tag5Compute,
                           test_databox_tags::Lambda0Compute,
                           test_databox_tags::Lambda1Compute>>(
        3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
    // [create_databox]
    return original_box;
  };
  {
    const auto original_box = create_original_box();
    static_assert(
        std::is_same<
            decltype(original_box),
            const db::DataBox<db::detail::expand_subitems<tmpl::append<
                tmpl::list<test_databox_tags::Tag0, test_databox_tags::Tag1,
                           test_databox_tags::Tag2>,
                tmpl::list<test_databox_tags::Tag4Compute,
                           test_databox_tags::Tag5Compute,
                           test_databox_tags::Lambda0Compute,
                           test_databox_tags::Lambda1Compute>>>>>::value,
        "Failed to create original_box");

    // [using_db_get]
    const auto& tag0 = db::get<test_databox_tags::Tag0>(original_box);
    // [using_db_get]
    CHECK(tag0 == 3.14);
    // Check retrieving chained compute item result
    CHECK(db::get<test_databox_tags::Tag5>(original_box) ==
          "My Sample String6.28"s);
    CHECK(db::get<test_databox_tags::Lambda0>(original_box) == 3.0 * 3.14);
    CHECK(db::get<test_databox_tags::Lambda1>(original_box) == 7.0);
  }
  // No removal
  {
    auto original_box = create_original_box();
    const auto& box =
        db::create_from<db::RemoveTags<>>(std::move(original_box));
    CHECK(db::get<test_databox_tags::Tag2>(box) == "My Sample String"s);
    CHECK(db::get<test_databox_tags::Tag5>(box) == "My Sample String6.28"s);
    CHECK(db::get<test_databox_tags::Lambda0>(box) == 3.0 * 3.14);
  }
  {
    // [create_from_remove]
    auto original_box = create_original_box();
    const auto& box = db::create_from<db::RemoveTags<test_databox_tags::Tag1>>(
        std::move(original_box));
    // [create_from_remove]
    CHECK(db::get<test_databox_tags::Tag2>(box) == "My Sample String"s);
    CHECK(db::get<test_databox_tags::Tag5>(box) == "My Sample String6.28"s);
    CHECK(db::get<test_databox_tags::Lambda0>(box) == 3.0 * 3.14);
  }
  {
    // [create_from_add_item]
    auto original_box = create_original_box();
    const auto& box =
        db::create_from<db::RemoveTags<>,
                        db::AddSimpleTags<test_databox_tags::Tag3>>(
            std::move(original_box), "Yet another test string"s);
    // [create_from_add_item]
    CHECK(db::get<test_databox_tags::Tag3>(box) == "Yet another test string"s);
    CHECK(db::get<test_databox_tags::Tag2>(box) == "My Sample String"s);
    // Check retrieving compute item result
    CHECK(db::get<test_databox_tags::Tag5>(box) == "My Sample String6.28"s);
    CHECK(db::get<test_databox_tags::Lambda0>(box) == 3.0 * 3.14);
  }
  {
    // [create_from_add_compute_item]
    auto simple_box = db::create<
        db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                          test_databox_tags::Tag2>>(
        3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
    const auto& box =
        db::create_from<db::RemoveTags<>, db::AddSimpleTags<>,
                        db::AddComputeTags<test_databox_tags::Tag4Compute>>(
            std::move(simple_box));
    // [create_from_add_compute_item]
    CHECK(db::get<test_databox_tags::Tag2>(box) == "My Sample String"s);
    // Check retrieving compute item result
    CHECK(db::get<test_databox_tags::Tag4>(box) == 6.28);
  }
  {
    auto simple_box = db::create<
        db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                          test_databox_tags::Tag2>>(
        3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
    const auto& box =
        db::create_from<db::RemoveTags<>,
                        db::AddSimpleTags<test_databox_tags::Tag3>,
                        db::AddComputeTags<test_databox_tags::Tag4Compute>>(
            std::move(simple_box), "Yet another test string"s);
    CHECK(db::get<test_databox_tags::Tag3>(box) == "Yet another test string"s);
    CHECK(db::get<test_databox_tags::Tag2>(box) == "My Sample String"s);
    // Check retrieving compute item result
    CHECK(db::get<test_databox_tags::Tag4>(box) == 6.28);
  }
  {
    auto simple_box = db::create<
        db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                          test_databox_tags::Tag2>>(
        3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
    const auto& box =
        db::create_from<db::RemoveTags<test_databox_tags::Tag1>,
                        db::AddSimpleTags<test_databox_tags::Tag3>,
                        db::AddComputeTags<test_databox_tags::Tag4Compute>>(
            std::move(simple_box), "Yet another test string"s);
    CHECK(db::get<test_databox_tags::Tag3>(box) == "Yet another test string"s);
    CHECK(db::get<test_databox_tags::Tag2>(box) == "My Sample String"s);
    // Check retrieving compute item result
    CHECK(6.28 == db::get<test_databox_tags::Tag4>(box));
  }
  {
    const auto box = db::create<
        db::AddSimpleTags<test_databox_tags::Pointer>,
        db::AddComputeTags<test_databox_tags::PointerToCounterCompute,
                           test_databox_tags::PointerToSumCompute>>(
        std::make_unique<int>(3));
    using DbTags = decltype(box)::tags_list;
    static_assert(
        std::is_same_v<
            db::detail::const_item_type<test_databox_tags::Pointer, DbTags>,
            const int&>,
        "Wrong type for const_item_type on unique_ptr simple item");
    static_assert(
        std::is_same_v<
            decltype(db::get<test_databox_tags::Pointer>(box)),
            db::detail::const_item_type<test_databox_tags::Pointer, DbTags>>,
        "Wrong type for get on unique_ptr simple item");
    CHECK(db::get<test_databox_tags::Pointer>(box) == 3);

    static_assert(
        std::is_same_v<
            db::detail::const_item_type<test_databox_tags::PointerBase, DbTags>,
            const int&>,
        "Wrong type for const_item_type on unique_ptr simple item by base");
    static_assert(
        std::is_same_v<decltype(db::get<test_databox_tags::PointerBase>(box)),
                       db::detail::const_item_type<
                           test_databox_tags::PointerBase, DbTags>>,
        "Wrong type for get on unique_ptr simple item by base");
    CHECK(db::get<test_databox_tags::PointerBase>(box) == 3);

    static_assert(
        std::is_same_v<db::detail::const_item_type<
                           test_databox_tags::PointerToCounter, DbTags>,
                       const int&>,
        "Wrong type for const_item_type on unique_ptr compute item");
    static_assert(
        std::is_same_v<
            decltype(db::get<test_databox_tags::PointerToCounter>(box)),
            db::detail::const_item_type<test_databox_tags::PointerToCounter,
                                        DbTags>>,
        "Wrong type for get on unique_ptr compute item");
    CHECK(db::get<test_databox_tags::PointerToCounter>(box) == 4);

    static_assert(
        std::is_same_v<db::detail::const_item_type<
                           test_databox_tags::PointerToCounterBase, DbTags>,
                       const int&>,
        "Wrong type for const_item_type on unique_ptr compute item by base");
    static_assert(
        std::is_same_v<
            decltype(db::get<test_databox_tags::PointerToCounterBase>(box)),
            db::detail::const_item_type<test_databox_tags::PointerToCounterBase,
                                        DbTags>>,
        "Wrong type for get on unique_ptr compute item by base");
    CHECK(db::get<test_databox_tags::PointerToCounterBase>(box) == 4);

    static_assert(std::is_same_v<db::detail::const_item_type<
                                     test_databox_tags::PointerToSum, DbTags>,
                                 const int&>,
                  "Wrong type for const_item_type on unique_ptr");
    static_assert(
        std::is_same_v<decltype(db::get<test_databox_tags::PointerToSum>(box)),
                       db::detail::const_item_type<
                           test_databox_tags::PointerToSum, DbTags>>,
        "Wrong type for get on unique_ptr");
    CHECK(db::get<test_databox_tags::PointerToSum>(box) == 8);
  }
}

namespace ArgumentTypeTags {
struct NonCopyable : db::SimpleTag {
  using type = ::NonCopyable;
};
template <size_t N>
struct String : db::SimpleTag {
  using type = std::string;
};
}  // namespace ArgumentTypeTags

void test_create_argument_types() noexcept {
  INFO("test create argument types");
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

void test_get_databox() noexcept {
  INFO("test get databox");
  auto original_box = db::create<
      db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                        test_databox_tags::Tag2>,
      db::AddComputeTags<test_databox_tags::Tag4Compute,
                         test_databox_tags::Tag5Compute>>(
      3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
  CHECK(std::addressof(original_box) ==
        std::addressof(db::get<Tags::DataBox>(original_box)));
  // [databox_self_tag_example]
  auto check_result_no_args = [](const auto& box) {
    CHECK(db::get<test_databox_tags::Tag2>(box) == "My Sample String"s);
    CHECK(db::get<test_databox_tags::Tag5>(box) == "My Sample String6.28"s);
  };
  db::apply<tmpl::list<Tags::DataBox>>(check_result_no_args, original_box);
  // [databox_self_tag_example]
}
}  // namespace

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
      db::AddComputeTags<test_databox_tags::Tag4Compute,
                         test_databox_tags::Tag5Compute>>(
      3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
  CHECK(std::addressof(original_box) ==
        std::addressof(db::get<Tags::DataBox>(original_box)));
  db::mutate<test_databox_tags::Tag0>(
      make_not_null(&original_box),
      [&original_box](const gsl::not_null<double*> /*tag0*/) {
        (void)db::get<Tags::DataBox>(original_box);
      });
}

namespace {
void test_mutate() noexcept {
  INFO("test mutate");
  auto original_box = db::create<
      db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                        test_databox_tags::Tag2, test_databox_tags::Pointer>,
      db::AddComputeTags<test_databox_tags::Tag4Compute,
                         test_databox_tags::Tag5Compute>>(
      3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s,
      std::make_unique<int>(3));
  CHECK(approx(db::get<test_databox_tags::Tag4>(original_box)) == 3.14 * 2.0);
  // [databox_mutate_example]
  db::mutate<test_databox_tags::Tag0, test_databox_tags::Tag1>(
      make_not_null(&original_box),
      [](const gsl::not_null<double*> tag0,
         const gsl::not_null<std::vector<double>*> tag1,
         const double compute_tag0) {
        CHECK(6.28 == compute_tag0);
        *tag0 = 10.32;
        (*tag1)[0] = 837.2;
      },
      db::get<test_databox_tags::Tag4>(original_box));
  CHECK(10.32 == db::get<test_databox_tags::Tag0>(original_box));
  CHECK(837.2 == db::get<test_databox_tags::Tag1>(original_box)[0]);
  // [databox_mutate_example]
  CHECK(approx(db::get<test_databox_tags::Tag4>(original_box)) == 10.32 * 2.0);

  auto result = db::mutate<test_databox_tags::Tag0>(
      make_not_null(&original_box),
      [](const gsl::not_null<double*> tag0, const double compute_tag0) {
        return *tag0 * compute_tag0;
      },
      db::get<test_databox_tags::Tag4>(original_box));
  CHECK(result == square(10.32) * 2.0);
  auto pointer_to_result = db::mutate<test_databox_tags::Tag0>(
      make_not_null(&original_box),
      [&result](const gsl::not_null<double*> tag0, const double compute_tag0) {
        *tag0 /= compute_tag0;
        return &result;
      },
      db::get<test_databox_tags::Tag4>(original_box));
  CHECK(db::get<test_databox_tags::Tag0>(original_box) == 0.5);
  CHECK(pointer_to_result == &result);

  db::mutate<test_databox_tags::Pointer>(
      make_not_null(&original_box), [](auto p) noexcept {
        static_assert(std::is_same_v<typename test_databox_tags::Pointer::type,
                                     std::unique_ptr<int>>,
                      "Wrong type for item_type on unique_ptr");
        static_assert(
            std::is_same_v<
                decltype(p),
                gsl::not_null<typename test_databox_tags::Pointer::type*>>,
            "Wrong type for mutate on unique_ptr");
        CHECK(**p == 3);
        *p = std::make_unique<int>(5);
      });
  CHECK(db::get<test_databox_tags::Pointer>(original_box) == 5);
}
}  // namespace

// [[OutputRegex, Unable to retrieve a \(compute\) item 'Tag4' from the
// DataBox from within a call to mutate. You must pass these either through the
// capture list of the lambda or the constructor of a class, this restriction
// exists to avoid complexity]]
SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.mutate_locked_get",
                  "[Unit][DataStructures]") {
  ERROR_TEST();
  auto original_box = db::create<
      db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                        test_databox_tags::Tag2>,
      db::AddComputeTags<test_databox_tags::Tag4Compute,
                         test_databox_tags::Tag5Compute>>(
      3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
  db::mutate<test_databox_tags::Tag0, test_databox_tags::Tag1>(
      make_not_null(&original_box),
      [&original_box](const gsl::not_null<double*> tag0,
                      const gsl::not_null<std::vector<double>*> tag1) {
        db::get<test_databox_tags::Tag4>(original_box);
        *tag0 = 10.32;
        (*tag1)[0] = 837.2;
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
      db::AddComputeTags<test_databox_tags::Tag4Compute,
                         test_databox_tags::Tag5Compute>>(
      3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
  db::mutate<test_databox_tags::Tag0>(
      make_not_null(&original_box),
      [&original_box](const gsl::not_null<double*> /*unused*/) {
        db::mutate<test_databox_tags::Tag1>(
            make_not_null(&original_box),
            [](const gsl::not_null<std::vector<double>*> tag1) {
              (*tag1)[0] = 10.0;
            });
      });
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

void test_apply() noexcept {
  INFO("test apply");
  auto original_box = db::create<
      db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                        test_databox_tags::Tag2, test_databox_tags::Pointer>,
      db::AddComputeTags<test_databox_tags::Tag4Compute,
                         test_databox_tags::Tag5Compute,
                         test_databox_tags::PointerToCounterCompute,
                         test_databox_tags::PointerToSumCompute>>(
      3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s,
      std::make_unique<int>(3));
  auto check_result_no_args = [](const std::string& sample_string,
                                 const auto& computed_string) {
    CHECK(sample_string == "My Sample String"s);
    CHECK(computed_string == "My Sample String6.28"s);
  };
  db::apply<tmpl::list<test_databox_tags::Tag2, test_databox_tags::Tag5>>(
      check_result_no_args, original_box);
  db::apply<tmpl::list<test_databox_tags::Tag2Base, test_databox_tags::Tag5>>(
      check_result_no_args, original_box);

  // [apply_example]
  auto check_result_args = [](const std::string& sample_string,
                              const auto& computed_string, const auto& vector) {
    CHECK(sample_string == "My Sample String"s);
    CHECK(computed_string == "My Sample String6.28"s);
    CHECK(vector == (std::vector<double>{8.7, 93.2, 84.7}));
  };
  db::apply<tmpl::list<test_databox_tags::Tag2, test_databox_tags::Tag5>>(
      check_result_args, original_box,
      db::get<test_databox_tags::Tag1>(original_box));
  // [apply_example]

  db::apply<tmpl::list<>>(NonCopyableFunctor{}, original_box);

  // [apply_struct_example]
  struct ApplyCallable {
    static void apply(const std::string& sample_string,
                      const std::string& computed_string,
                      const std::vector<double>& vector) noexcept {
      CHECK(sample_string == "My Sample String"s);
      CHECK(computed_string == "My Sample String6.28"s);
      CHECK(vector == (std::vector<double>{8.7, 93.2, 84.7}));
    }
  };
  db::apply<tmpl::list<test_databox_tags::Tag2, test_databox_tags::Tag5>>(
      ApplyCallable{}, original_box,
      db::get<test_databox_tags::Tag1>(original_box));
  // [apply_struct_example]
  db::apply<tmpl::list<test_databox_tags::Tag2Base, test_databox_tags::Tag5>>(
      ApplyCallable{}, original_box,
      db::get<test_databox_tags::Tag1>(original_box));
  // [apply_stateless_struct_example]
  struct StatelessApplyCallable {
    using argument_tags =
        tmpl::list<test_databox_tags::Tag2, test_databox_tags::Tag5>;
    static void apply(const std::string& sample_string,
                      const std::string& computed_string,
                      const std::vector<double>& vector) noexcept {
      CHECK(sample_string == "My Sample String"s);
      CHECK(computed_string == "My Sample String6.28"s);
      CHECK(vector == (std::vector<double>{8.7, 93.2, 84.7}));
    }
  };
  db::apply<StatelessApplyCallable>(
      original_box, db::get<test_databox_tags::Tag1>(original_box));
  // [apply_stateless_struct_example]
  db::apply(StatelessApplyCallable{}, original_box,
            db::get<test_databox_tags::Tag1>(original_box));

  db::apply<tmpl::list<
      test_databox_tags::Pointer, test_databox_tags::PointerToCounter,
      test_databox_tags::PointerToSum, test_databox_tags::PointerBase,
      test_databox_tags::PointerToCounterBase>>(
      [](const int& simple, const int& compute, const int& compute_mutating,
         const int& simple_base, const int& compute_base) noexcept {
        CHECK(simple == 3);
        CHECK(simple_base == 3);
        CHECK(compute == 4);
        CHECK(compute_base == 4);
        CHECK(compute_mutating == 8);
      },
      original_box);

  struct PointerApplyCallable {
    using argument_tags = tmpl::list<
        test_databox_tags::Pointer, test_databox_tags::PointerToCounter,
        test_databox_tags::PointerToSum, test_databox_tags::PointerBase,
        test_databox_tags::PointerToCounterBase>;
    static void apply(const int& simple, const int& compute,
                      const int& compute_mutating, const int& simple_base,
                      const int& compute_base) noexcept {
      CHECK(simple == 3);
      CHECK(simple_base == 3);
      CHECK(compute == 4);
      CHECK(compute_base == 4);
      CHECK(compute_mutating == 8);
    }
  };
  db::apply<PointerApplyCallable>(original_box);

  {
    INFO("Test apply with optional reference argument");
    db::apply<tmpl::list<test_databox_tags::Tag1, test_databox_tags::Tag1>>(
        [](const std::optional<
               std::reference_wrapper<const std::vector<double>>>
               optional_ref,
           const auto& const_ref) {
          REQUIRE(optional_ref.has_value());
          CHECK(optional_ref->get() == const_ref);
        },
        original_box);
  }
}

struct Var1 : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame::Grid>;
};

struct Var1Compute : Var1, db::ComputeTag {
  using base = Var1;
  using return_type = tnsr::I<DataVector, 3, Frame::Grid>;
  static void function(const gsl::not_null<return_type*> result) {
    *result = tnsr::I<DataVector, 3, Frame::Grid>(5_st, 2.0);
  }
  using argument_tags = tmpl::list<>;
};

struct Var2 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <class Tag, class VolumeDim, class Frame>
struct PrefixTag0 : db::PrefixTag, db::SimpleTag {
  using type = TensorMetafunctions::prepend_spatial_index<
      typename Tag::type, VolumeDim::value, UpLo::Lo, Frame>;
  using tag = Tag;
};

using two_vars = tmpl::list<Var1, Var2>;
using vector_only = tmpl::list<Var1>;
using scalar_only = tmpl::list<Var2>;

static_assert(
    std::is_same_v<
        tmpl::back<db::wrap_tags_in<PrefixTag0, scalar_only, tmpl::size_t<2>,
                                    Frame::Grid>>::type,
        tnsr::i<DataVector, 2, Frame::Grid>>,
    "Failed db::wrap_tags_in scalar_only");

static_assert(
    std::is_same_v<
        tmpl::back<db::wrap_tags_in<PrefixTag0, vector_only, tmpl::size_t<3>,
                                    Frame::Grid>>::type,
        tnsr::iJ<DataVector, 3, Frame::Grid>>,
    "Failed db::wrap_tags_in vector_only");

static_assert(
    std::is_same_v<
        tmpl::back<db::wrap_tags_in<PrefixTag0, two_vars, tmpl::size_t<2>,
                                    Frame::Grid>>::type,
        tnsr::i<DataVector, 2, Frame::Grid>>,
    "Failed db::wrap_tags_in two_vars scalar");

static_assert(
    std::is_same_v<
        tmpl::front<db::wrap_tags_in<PrefixTag0, two_vars, tmpl::size_t<3>,
                                     Frame::Grid>>::type,
        tnsr::iJ<DataVector, 3, Frame::Grid>>,
    "Failed db::wrap_tags_in two_vars vector");

namespace test_databox_tags {
struct ScalarTagBase : db::BaseTag {};
struct ScalarTag : db::SimpleTag, ScalarTagBase {
  using type = Scalar<DataVector>;
};
struct VectorTag : db::SimpleTag {
  using type = tnsr::I<DataVector, 3>;
};
struct ScalarTag2 : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct VectorTag2 : db::SimpleTag {
  using type = tnsr::I<DataVector, 3>;
};
struct ScalarTag3 : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct VectorTag3 : db::SimpleTag {
  using type = tnsr::I<DataVector, 3>;
};
struct ScalarTag4 : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct VectorTag4 : db::SimpleTag {
  using type = tnsr::I<DataVector, 3>;
};
struct IncompleteType; // Forward declare, do not define on purpose.
struct TagForIncompleteType : db::SimpleTag {
  using type = IncompleteType;
};
}  // namespace test_databox_tags

namespace {
void multiply_scalar_by_two(
    const gsl::not_null<Variables<tmpl::list<test_databox_tags::ScalarTag2,
                                             test_databox_tags::VectorTag2>>*>
        result,
    const Scalar<DataVector>& scalar) {
  *result = Variables<
      tmpl::list<test_databox_tags::ScalarTag2, test_databox_tags::VectorTag2>>{
      scalar.begin()->size(), 2.0};
  get<test_databox_tags::ScalarTag2>(*result).get() = scalar.get() * 2.0;
}

void multiply_scalar_by_four(const gsl::not_null<Scalar<DataVector>*> result,
                             const Scalar<DataVector>& scalar) {
  *result = Scalar<DataVector>(scalar.get() * 4.0);
}

void multiply_scalar_by_three(const gsl::not_null<Scalar<DataVector>*> result,
                              const Scalar<DataVector>& scalar) {
  *result = Scalar<DataVector>(scalar.get() * 3.0);
}

void divide_scalar_by_three(const gsl::not_null<Scalar<DataVector>*> result,
                            const Scalar<DataVector>& scalar) {
  *result = Scalar<DataVector>(scalar.get() / 3.0);
}

void divide_scalar_by_two(
    const gsl::not_null<Variables<tmpl::list<test_databox_tags::VectorTag3,
                                             test_databox_tags::ScalarTag3>>*>
        result,
    const Scalar<DataVector>& scalar) {
  *result = Variables<
      tmpl::list<test_databox_tags::VectorTag3, test_databox_tags::ScalarTag3>>{
      scalar.begin()->size(), 10.0};
  get<test_databox_tags::ScalarTag3>(*result).get() = scalar.get() / 2.0;
}

void multiply_variables_by_two(
    const gsl::not_null<Variables<tmpl::list<test_databox_tags::ScalarTag4,
                                             test_databox_tags::VectorTag4>>*>
        result,
    const Variables<tmpl::list<test_databox_tags::ScalarTag,
                               test_databox_tags::VectorTag>>& vars) {
  *result = Variables<
      tmpl::list<test_databox_tags::ScalarTag4, test_databox_tags::VectorTag4>>(
      vars.number_of_grid_points(), 2.0);
  get<test_databox_tags::ScalarTag4>(*result).get() *=
      get<test_databox_tags::ScalarTag>(vars).get();
  get<0>(get<test_databox_tags::VectorTag4>(*result)) *=
      get<0>(get<test_databox_tags::VectorTag>(vars));
  get<1>(get<test_databox_tags::VectorTag4>(*result)) *=
      get<1>(get<test_databox_tags::VectorTag>(vars));
  get<2>(get<test_databox_tags::VectorTag4>(*result)) *=
      get<2>(get<test_databox_tags::VectorTag>(vars));
}
}  // namespace

namespace test_databox_tags {
struct MultiplyScalarByTwo : db::SimpleTag {
  using type = Variables<
      tmpl::list<test_databox_tags::ScalarTag2, test_databox_tags::VectorTag2>>;
};

struct MultiplyScalarByTwoCompute : MultiplyScalarByTwo, db::ComputeTag {
  using base = MultiplyScalarByTwo;
  using variables_tags =
      tmpl::list<test_databox_tags::ScalarTag2, test_databox_tags::VectorTag2>;
  using return_type = Variables<variables_tags>;
  static constexpr auto function = multiply_scalar_by_two;
  using argument_tags = tmpl::list<test_databox_tags::ScalarTag>;
};

struct MultiplyScalarByFour : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct MultiplyScalarByFourCompute : MultiplyScalarByFour, db::ComputeTag {
  using base = MultiplyScalarByFour;
  using return_type = Scalar<DataVector>;
  static constexpr auto function = multiply_scalar_by_four;
  using argument_tags = tmpl::list<test_databox_tags::ScalarTag2>;
};

struct MultiplyScalarByThree : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct MultiplyScalarByThreeCompute : MultiplyScalarByThree, db::ComputeTag {
  using base = MultiplyScalarByThree;
  using return_type = Scalar<DataVector>;
  static constexpr auto function = multiply_scalar_by_three;
  using argument_tags = tmpl::list<test_databox_tags::MultiplyScalarByFour>;
};

struct DivideScalarByThree : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct DivideScalarByThreeCompute : DivideScalarByThree, db::ComputeTag {
  using base = DivideScalarByThree;
  using return_type = Scalar<DataVector>;
  static constexpr auto function = divide_scalar_by_three;
  using argument_tags = tmpl::list<test_databox_tags::MultiplyScalarByThree>;
};

struct DivideScalarByTwo : db::SimpleTag {
  using type = Variables<
      tmpl::list<test_databox_tags::VectorTag3, test_databox_tags::ScalarTag3>>;
};

struct DivideScalarByTwoCompute : DivideScalarByTwo, db::ComputeTag {
  using base = DivideScalarByTwo;
  using return_type = Variables<
      tmpl::list<test_databox_tags::VectorTag3, test_databox_tags::ScalarTag3>>;
  static constexpr auto function = divide_scalar_by_two;
  using argument_tags = tmpl::list<test_databox_tags::DivideScalarByThree>;
};

struct MultiplyVariablesByTwo : db::SimpleTag {
  using type = Variables<
      tmpl::list<test_databox_tags::ScalarTag4, test_databox_tags::VectorTag4>>;
};

struct MultiplyVariablesByTwoCompute : MultiplyVariablesByTwo, db::ComputeTag {
  using base = MultiplyVariablesByTwo;
  using return_type = Variables<
      tmpl::list<test_databox_tags::ScalarTag4, test_databox_tags::VectorTag4>>;
  static constexpr auto function = multiply_variables_by_two;
  using argument_tags = tmpl::list<Tags::Variables<
      tmpl::list<test_databox_tags::ScalarTag, test_databox_tags::VectorTag>>>;
};
}  // namespace test_databox_tags

void test_variables() noexcept {
  INFO("test variables");
  using vars_tag = Tags::Variables<
      tmpl::list<test_databox_tags::ScalarTag, test_databox_tags::VectorTag>>;
  auto box = db::create<
      db::AddSimpleTags<vars_tag>,
      db::AddComputeTags<test_databox_tags::MultiplyScalarByTwoCompute,
                         test_databox_tags::MultiplyScalarByFourCompute,
                         test_databox_tags::MultiplyScalarByThreeCompute,
                         test_databox_tags::DivideScalarByThreeCompute,
                         test_databox_tags::DivideScalarByTwoCompute,
                         test_databox_tags::MultiplyVariablesByTwoCompute>>(
      Variables<tmpl::list<test_databox_tags::ScalarTag,
                           test_databox_tags::VectorTag>>(2, 3.));
  const auto check_references_match = [&box]() noexcept {
    const auto& vars_original = db::get<vars_tag>(box);
    CHECK(get(db::get<test_databox_tags::ScalarTag>(box)).data() ==
          get(get<test_databox_tags::ScalarTag>(vars_original)).data());
    for (size_t i = 0; i < 3; ++i) {
      CHECK(db::get<test_databox_tags::VectorTag>(box).get(i).data() ==
            get<test_databox_tags::VectorTag>(vars_original).get(i).data());
    }

    const auto& vars_mul_scalar =
        db::get<test_databox_tags::MultiplyScalarByTwo>(box);
    CHECK(get(db::get<test_databox_tags::ScalarTag2>(box)).data() ==
          get(get<test_databox_tags::ScalarTag2>(vars_mul_scalar)).data());
    for (size_t i = 0; i < 3; ++i) {
      CHECK(db::get<test_databox_tags::VectorTag2>(box).get(i).data() ==
            get<test_databox_tags::VectorTag2>(vars_mul_scalar).get(i).data());
    }

    const auto& vars_div_scalar =
        db::get<test_databox_tags::DivideScalarByTwo>(box);
    CHECK(get(db::get<test_databox_tags::ScalarTag3>(box)).data() ==
          get(get<test_databox_tags::ScalarTag3>(vars_div_scalar)).data());
    for (size_t i = 0; i < 3; ++i) {
      CHECK(db::get<test_databox_tags::VectorTag3>(box).get(i).data() ==
            get<test_databox_tags::VectorTag3>(vars_div_scalar).get(i).data());
    }

    const auto& vars_mul_two =
        db::get<test_databox_tags::MultiplyVariablesByTwo>(box);
    CHECK(get(db::get<test_databox_tags::ScalarTag4>(box)).data() ==
          get(get<test_databox_tags::ScalarTag4>(vars_mul_two)).data());
    for (size_t i = 0; i < 3; ++i) {
      CHECK(db::get<test_databox_tags::VectorTag4>(box).get(i).data() ==
            get<test_databox_tags::VectorTag4>(vars_mul_two).get(i).data());
    }
  };

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
  check_references_match();

  db::mutate<test_databox_tags::ScalarTag>(
      make_not_null(&box), [](const gsl::not_null<Scalar<DataVector>*> scalar) {
        scalar->get() = 4.0;
      });

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
  check_references_match();

  db::mutate<Tags::Variables<
      tmpl::list<test_databox_tags::ScalarTag, test_databox_tags::VectorTag>>>(
      make_not_null(&box), [](const auto vars) {
        get<test_databox_tags::ScalarTag>(*vars).get() = 6.0;
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
  check_references_match();

  db::mutate<Tags::Variables<
      tmpl::list<test_databox_tags::ScalarTag, test_databox_tags::VectorTag>>>(
      make_not_null(&box), [](const auto vars) {
        get<test_databox_tags::ScalarTag>(*vars).get() = 4.0;
        get<test_databox_tags::VectorTag>(*vars) =
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
  check_references_match();
}

struct Tag1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct Tag2 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

void test_variables2() noexcept {
  INFO("test variables2");
  auto box =
      db::create<db::AddSimpleTags<Tags::Variables<tmpl::list<Tag1, Tag2>>>>(
          Variables<tmpl::list<Tag1, Tag2>>(1, 1.));

  db::mutate<Tags::Variables<tmpl::list<Tag1, Tag2>>>(
      make_not_null(&box), [](const auto vars) {
        *vars = Variables<tmpl::list<Tag1, Tag2>>(1, 2.);
      });
  CHECK(db::get<Tag1>(box) == Scalar<DataVector>(DataVector(1, 2.)));
}

void test_reset_compute_items() noexcept {
  INFO("test reset compute items");
  auto box = db::create<
      db::AddSimpleTags<
          test_databox_tags::Tag0, test_databox_tags::Tag1,
          test_databox_tags::Tag2,
          Tags::Variables<tmpl::list<test_databox_tags::ScalarTag,
                                     test_databox_tags::VectorTag>>>,
      db::AddComputeTags<test_databox_tags::Tag4Compute,
                         test_databox_tags::Tag5Compute,
                         test_databox_tags::MultiplyScalarByTwoCompute,
                         test_databox_tags::MultiplyVariablesByTwoCompute>>(
      3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s,
      Variables<tmpl::list<test_databox_tags::ScalarTag,
                           test_databox_tags::VectorTag>>(2, 3.));
  CHECK(approx(db::get<test_databox_tags::Tag4>(box)) == 3.14 * 2.0);
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
}

namespace ExtraResetTags {
struct Var : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct Int : db::SimpleTag {
  using type = int;
};
struct CheckReset : db::SimpleTag {
  using type = int;
};
struct CheckResetCompute : CheckReset, db::ComputeTag {
  using base = CheckReset;
  using return_type = int;
  static auto function(
      const gsl::not_null<int*> result,
      const ::Variables<tmpl::list<Var>>& /*unused*/) noexcept {
    static bool first_call = true;
    CHECK(first_call);
    first_call = false;
    *result = 0;
  }
  using argument_tags = tmpl::list<Tags::Variables<tmpl::list<Var>>>;
};
}  // namespace ExtraResetTags

void test_variables_extra_reset() noexcept {
  INFO("test variables extra reset");
  auto box = db::create<
      db::AddSimpleTags<ExtraResetTags::Int,
                        Tags::Variables<tmpl::list<ExtraResetTags::Var>>>,
      db::AddComputeTags<ExtraResetTags::CheckResetCompute>>(
      1, Variables<tmpl::list<ExtraResetTags::Var>>(2, 3.));
  CHECK(db::get<ExtraResetTags::CheckReset>(box) == 0);
  db::mutate<ExtraResetTags::Int>(make_not_null(&box),
                                  [](const gsl::not_null<int*> /*unused*/) {});
  CHECK(db::get<ExtraResetTags::CheckReset>(box) == 0);
}

// [mutate_apply_struct_definition_example]
struct TestDataboxMutateApply {
  // delete copy semantics just to make sure it works. Not necessary in general.
  TestDataboxMutateApply() = default;
  TestDataboxMutateApply(const TestDataboxMutateApply&) = delete;
  TestDataboxMutateApply& operator=(const TestDataboxMutateApply&) = delete;
  TestDataboxMutateApply(TestDataboxMutateApply&&) = default;
  TestDataboxMutateApply& operator=(TestDataboxMutateApply&&) = default;
  ~TestDataboxMutateApply() = default;

  // These typelists are used by the `db::mutate_apply` overload that does not
  // require these lists as template arguments
  using return_tags =
      tmpl::list<test_databox_tags::ScalarTag, test_databox_tags::VectorTag>;
  using argument_tags = tmpl::list<test_databox_tags::Tag2>;

  static void apply(const gsl::not_null<Scalar<DataVector>*> scalar,
                    const gsl::not_null<tnsr::I<DataVector, 3>*> vector,
                    const std::string& tag2) noexcept {
    scalar->get() *= 2.0;
    get<0>(*vector) *= 3.0;
    get<1>(*vector) *= 4.0;
    get<2>(*vector) *= 5.0;
    CHECK(tag2 == "My Sample String"s);
  }
};
// [mutate_apply_struct_definition_example]

struct TestDataboxMutateApplyBase {
  using return_tags = tmpl::list<test_databox_tags::ScalarTag>;
  using argument_tags = tmpl::list<test_databox_tags::Tag2Base>;

  static void apply(const gsl::not_null<Scalar<DataVector>*> scalar,
                    const std::string& tag2) noexcept {
    CHECK(*scalar == Scalar<DataVector>(DataVector(2, 6.)));
    CHECK(tag2 == "My Sample String"s);
  }
};

struct PointerMutateApply {
  using return_tags = tmpl::list<test_databox_tags::Pointer>;
  using argument_tags = tmpl::list<test_databox_tags::PointerToCounter,
                                   test_databox_tags::PointerToSum>;
  static void apply(const gsl::not_null<std::unique_ptr<int>*> ret,
                    const int& compute, const int& compute_mutating) noexcept {
    **ret = 7;
    CHECK(compute == 7);
    CHECK(compute_mutating == 14);
  }
};

struct PointerMutateApplyBase {
  using return_tags = tmpl::list<test_databox_tags::Pointer>;
  using argument_tags = tmpl::list<test_databox_tags::PointerToCounterBase>;
  static void apply(const gsl::not_null<std::unique_ptr<int>*> ret,
                    const int& compute_base) noexcept {
    **ret = 8;
    CHECK(compute_base == 8);
  }
};

void test_mutate_apply() noexcept {
  INFO("test mutate apply");
  auto box = db::create<
      db::AddSimpleTags<
          test_databox_tags::Tag0, test_databox_tags::Tag1,
          test_databox_tags::Tag2,
          Tags::Variables<tmpl::list<test_databox_tags::ScalarTag,
                                     test_databox_tags::VectorTag>>,
          test_databox_tags::Pointer>,
      db::AddComputeTags<test_databox_tags::Tag4Compute,
                         test_databox_tags::Tag5Compute,
                         test_databox_tags::MultiplyScalarByTwoCompute,
                         test_databox_tags::PointerToCounterCompute,
                         test_databox_tags::PointerToSumCompute>>(
      3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s,
      Variables<tmpl::list<test_databox_tags::ScalarTag,
                           test_databox_tags::VectorTag>>(2, 3.),
      std::make_unique<int>(3));
  CHECK(approx(db::get<test_databox_tags::Tag4>(box)) == 3.14 * 2.0);
  CHECK(db::get<test_databox_tags::ScalarTag>(box) ==
        Scalar<DataVector>(DataVector(2, 3.)));
  CHECK(db::get<test_databox_tags::VectorTag>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 3.))));
  CHECK(db::get<test_databox_tags::ScalarTag2>(box) ==
        Scalar<DataVector>(DataVector(2, 6.)));
  CHECK(db::get<test_databox_tags::VectorTag2>(box) ==
        (tnsr::I<DataVector, 3>(DataVector(2, 2.))));

  {
    INFO("Apply function or lambda");
    // [mutate_apply_struct_example_stateful]
    db::mutate_apply(TestDataboxMutateApply{}, make_not_null(&box));
    // [mutate_apply_struct_example_stateful]
    db::mutate_apply(TestDataboxMutateApplyBase{}, make_not_null(&box));

    CHECK(approx(db::get<test_databox_tags::Tag4>(box)) == 3.14 * 2.0);
    CHECK(db::get<test_databox_tags::ScalarTag>(box) ==
          Scalar<DataVector>(DataVector(2, 6.)));
    CHECK(db::get<test_databox_tags::VectorTag>(box) ==
          (tnsr::I<DataVector, 3>{
              {{DataVector(2, 9.), DataVector(2, 12.), DataVector(2, 15.)}}}));
    CHECK(db::get<test_databox_tags::ScalarTag2>(box) ==
          Scalar<DataVector>(DataVector(2, 12.)));
    CHECK(db::get<test_databox_tags::VectorTag2>(box) ==
          (tnsr::I<DataVector, 3>(DataVector(2, 2.))));
    // [mutate_apply_lambda_example]
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
        make_not_null(&box));
    // [mutate_apply_lambda_example]
    db::mutate_apply<tmpl::list<test_databox_tags::ScalarTag>,
                     tmpl::list<test_databox_tags::Tag2Base>>(
        [](const gsl::not_null<Scalar<DataVector>*> scalar,
           const std::string& tag2) {
          CHECK(*scalar == Scalar<DataVector>(DataVector(2, 12.)));
          CHECK(tag2 == "My Sample String"s);
        },
        make_not_null(&box));
    CHECK(approx(db::get<test_databox_tags::Tag4>(box)) == 3.14 * 2.0);
    CHECK(db::get<test_databox_tags::ScalarTag>(box) ==
          Scalar<DataVector>(DataVector(2, 12.)));
    CHECK(db::get<test_databox_tags::VectorTag>(box) ==
          (tnsr::I<DataVector, 3>{
              {{DataVector(2, 27.), DataVector(2, 48.), DataVector(2, 75.)}}}));
    CHECK(db::get<test_databox_tags::ScalarTag2>(box) ==
          Scalar<DataVector>(DataVector(2, 24.)));
    CHECK(db::get<test_databox_tags::VectorTag2>(box) ==
          (tnsr::I<DataVector, 3>(DataVector(2, 2.))));
    // check with a forwarded return value
    size_t size_of_internal_string =
        db::mutate_apply<tmpl::list<test_databox_tags::ScalarTag>,
                         tmpl::list<test_databox_tags::Tag2Base>>(
            [](const gsl::not_null<Scalar<DataVector>*> scalar,
               const std::string& tag2) {
              CHECK(*scalar == Scalar<DataVector>(DataVector(2, 12.)));
              CHECK(tag2 == "My Sample String"s);
              return tag2.size();
            },
            make_not_null(&box));
    CHECK(size_of_internal_string == 16_st);

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
        make_not_null(&box));

    CHECK(approx(db::get<test_databox_tags::Tag4>(box)) == 3.14 * 2.0);
    CHECK(db::get<test_databox_tags::ScalarTag>(box) ==
          Scalar<DataVector>(DataVector(2, 24.)));
    CHECK(db::get<test_databox_tags::VectorTag>(box) ==
          (tnsr::I<DataVector, 3>{{{DataVector(2, 81.), DataVector(2, 192.),
                                    DataVector(2, 375.)}}}));
    CHECK(db::get<test_databox_tags::ScalarTag2>(box) ==
          Scalar<DataVector>(DataVector(2, 48.)));
    CHECK(db::get<test_databox_tags::VectorTag2>(box) ==
          (tnsr::I<DataVector, 3>(DataVector(2, 2.))));
  }

  {
    INFO("Stateless struct with tags lists");
    // [mutate_apply_struct_example_stateless]
    db::mutate_apply<TestDataboxMutateApply>(make_not_null(&box));
    // [mutate_apply_struct_example_stateless]
    CHECK(approx(db::get<test_databox_tags::Tag4>(box)) == 3.14 * 2.0);
    CHECK(db::get<test_databox_tags::ScalarTag>(box) ==
          Scalar<DataVector>(DataVector(2, 48.)));
    CHECK(db::get<test_databox_tags::VectorTag>(box) ==
          (tnsr::I<DataVector, 3>{{{DataVector(2, 243.), DataVector(2, 768.),
                                    DataVector(2, 1875.)}}}));
    CHECK(db::get<test_databox_tags::ScalarTag2>(box) ==
          Scalar<DataVector>(DataVector(2, 96.)));
    CHECK(db::get<test_databox_tags::VectorTag2>(box) ==
          (tnsr::I<DataVector, 3>(DataVector(2, 2.))));
  }

  {
    INFO("unique_ptr");
    db::mutate_apply<tmpl::list<test_databox_tags::Pointer>, tmpl::list<>>(
        [](const gsl::not_null<std::unique_ptr<int>*> p) noexcept { **p = 5; },
        make_not_null(&box));
    db::mutate_apply<tmpl::list<>,
                     tmpl::list<test_databox_tags::Pointer,
                                test_databox_tags::PointerToCounter,
                                test_databox_tags::PointerToSum>>(
        [](const int& simple, const int& compute,
           const int& compute_mutating) noexcept {
          CHECK(simple == 5);
          CHECK(compute == 6);
          CHECK(compute_mutating == 12);
        },
        make_not_null(&box));
    db::mutate_apply<tmpl::list<test_databox_tags::Pointer>, tmpl::list<>>(
        [](const gsl::not_null<std::unique_ptr<int>*> p) noexcept { **p = 6; },
        make_not_null(&box));
    db::mutate_apply<tmpl::list<>,
                     tmpl::list<test_databox_tags::PointerBase,
                                test_databox_tags::PointerToCounterBase>>(
        [](const int& simple_base, const int& compute_base) noexcept {
          CHECK(simple_base == 6);
          CHECK(compute_base == 7);
        },
        make_not_null(&box));

    db::mutate_apply<PointerMutateApply>(make_not_null(&box));

    db::mutate_apply<PointerMutateApplyBase>(make_not_null(&box));
    CHECK(db::get<test_databox_tags::Pointer>(box) == 8);
  }
}

static_assert(
    std::is_same_v<
        db::compute_databox_type<tmpl::list<
            test_databox_tags::Tag0, test_databox_tags::Tag1,
            Tags::Variables<tmpl::list<test_databox_tags::ScalarTag,
                                       test_databox_tags::VectorTag>>,
            test_databox_tags::Tag4Compute,
            test_databox_tags::MultiplyScalarByTwoCompute>>,
        db::DataBox<tmpl::list<
            test_databox_tags::Tag0, test_databox_tags::Tag1,
            Tags::Variables<brigand::list<test_databox_tags::ScalarTag,
                                          test_databox_tags::VectorTag>>,
            test_databox_tags::ScalarTag, test_databox_tags::VectorTag,
            test_databox_tags::Tag4Compute,
            test_databox_tags::MultiplyScalarByTwoCompute,
            ::Tags::Subitem<test_databox_tags::ScalarTag2,
                            test_databox_tags::MultiplyScalarByTwoCompute>,
            ::Tags::Subitem<test_databox_tags::VectorTag2,
                            test_databox_tags::MultiplyScalarByTwoCompute>>>>,
    "Failed testing db::compute_databox_type");

static_assert(
    std::is_same_v<
        db::compute_databox_type<
            tmpl::list<test_databox_tags::TagForIncompleteType>>,
        db::DataBox<tmpl::list<test_databox_tags::TagForIncompleteType>>>,
    "Failed testing db::compute_databox_type for incomplete type");

void multiply_by_two_mutate(const gsl::not_null<std::vector<double>*> t,
                            const double value) {
  if (t->empty()) {
    t->resize(10);
  }
  for (auto& p : *t) {
    p = 2.0 * value;
  }
}

// [databox_mutating_compute_item_function]
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
// [databox_mutating_compute_item_function]

namespace test_databox_tags {
struct MutateTag0 : db::SimpleTag {
  using type = std::vector<double>;
};

struct MutateTag0Compute : MutateTag0, db::ComputeTag {
  using return_type = std::vector<double>;
  using base = MutateTag0;
  static constexpr auto function = multiply_by_two_mutate;
  using argument_tags = tmpl::list<Tag0>;
};

struct MutateVariables : db::SimpleTag {
  using type = Variables<
      tmpl::list<test_databox_tags::ScalarTag, test_databox_tags::VectorTag>>;
};

// [databox_mutating_compute_item_tag]
struct MutateVariablesCompute : MutateVariables, db::ComputeTag {
  using base = MutateVariables;
  static constexpr auto function = mutate_variables;
  using return_type = Variables<
      tmpl::list<test_databox_tags::ScalarTag, test_databox_tags::VectorTag>>;
  using argument_tags = tmpl::list<Tag0>;
};
// [databox_mutating_compute_item_tag]
}  // namespace test_databox_tags

void test_mutating_compute_item() noexcept {
  INFO("test mutating compute item");
  auto original_box = db::create<
      db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag1,
                        test_databox_tags::Tag2>,
      db::AddComputeTags<test_databox_tags::MutateTag0Compute,
                         test_databox_tags::MutateVariablesCompute,
                         test_databox_tags::Tag4Compute,
                         test_databox_tags::Tag5Compute>>(
      3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s);
  const double* const initial_data_location_mutating =
      db::get<test_databox_tags::MutateTag0>(original_box).data();
  const std::array<const double* const, 4>
      initial_variables_data_location_mutate{
          {get<test_databox_tags::ScalarTag>(
               db::get<test_databox_tags::MutateVariables>(original_box))
               .get()
               .data(),
           get<0>(
               get<test_databox_tags::VectorTag>(
                   db::get<test_databox_tags::MutateVariables>(original_box)))
               .data(),
           get<1>(
               get<test_databox_tags::VectorTag>(
                   db::get<test_databox_tags::MutateVariables>(original_box)))
               .data(),
           get<2>(
               get<test_databox_tags::VectorTag>(
                   db::get<test_databox_tags::MutateVariables>(original_box)))
               .data()}};

  CHECK(approx(db::get<test_databox_tags::Tag4>(original_box)) == 3.14 * 2.0);
  CHECK_ITERABLE_APPROX(db::get<test_databox_tags::MutateTag0>(original_box),
                        std::vector<double>(10, 2.0 * 3.14));
  CHECK_ITERABLE_APPROX(
      get<test_databox_tags::ScalarTag>(
          db::get<test_databox_tags::MutateVariables>(original_box)),
      Scalar<DataVector>(DataVector(10, 2.0 * 3.14)));
  CHECK_ITERABLE_APPROX(
      get<test_databox_tags::VectorTag>(
          db::get<test_databox_tags::MutateVariables>(original_box)),
      typename test_databox_tags::VectorTag::type(DataVector(10, 3.0 * 3.14)));

  db::mutate<test_databox_tags::Tag0, test_databox_tags::Tag1>(
      make_not_null(&original_box),
      [](const gsl::not_null<double*> tag0,
         const gsl::not_null<std::vector<double>*> tag1,
         const double compute_tag0) {
        CHECK(6.28 == compute_tag0);
        *tag0 = 10.32;
        (*tag1)[0] = 837.2;
      },
      db::get<test_databox_tags::Tag4>(original_box));

  CHECK(10.32 == db::get<test_databox_tags::Tag0>(original_box));
  CHECK(837.2 == db::get<test_databox_tags::Tag1>(original_box)[0]);
  CHECK(approx(db::get<test_databox_tags::Tag4>(original_box)) == 10.32 * 2.0);
  CHECK_ITERABLE_APPROX(db::get<test_databox_tags::MutateTag0>(original_box),
                        std::vector<double>(10, 2.0 * 10.32));
  CHECK(initial_data_location_mutating ==
        db::get<test_databox_tags::MutateTag0>(original_box).data());
  CHECK_ITERABLE_APPROX(
      get<test_databox_tags::ScalarTag>(
          db::get<test_databox_tags::MutateVariables>(original_box)),
      Scalar<DataVector>(DataVector(10, 2.0 * 10.32)));
  CHECK_ITERABLE_APPROX(
      get<test_databox_tags::VectorTag>(
          db::get<test_databox_tags::MutateVariables>(original_box)),
      typename test_databox_tags::VectorTag::type(DataVector(10, 3.0 * 10.32)));

  // Check that the memory allocated by std::vector has not changed, which is
  // the key feature of mutating compute items.
  CHECK(initial_variables_data_location_mutate ==
        (std::array<const double* const, 4>{
            {get<test_databox_tags::ScalarTag>(
                 db::get<test_databox_tags::MutateVariables>(original_box))
                 .get()
                 .data(),
             get<0>(
                 get<test_databox_tags::VectorTag>(
                     db::get<test_databox_tags::MutateVariables>(original_box)))
                 .data(),
             get<1>(
                 get<test_databox_tags::VectorTag>(
                     db::get<test_databox_tags::MutateVariables>(original_box)))
                 .data(),
             get<2>(
                 get<test_databox_tags::VectorTag>(
                     db::get<test_databox_tags::MutateVariables>(original_box)))
                 .data()}}));
}

namespace DataBoxTest_detail {
struct vector : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame::Grid>;
};

struct scalar : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct vector2 : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame::Grid>;
};
}  // namespace DataBoxTest_detail

void test_data_on_slice_single() noexcept {
  INFO("test data on slice single");
  const size_t x_extents = 2;
  const size_t y_extents = 3;
  const size_t z_extents = 4;
  const size_t vec_size = DataBoxTest_detail::vector::type::size();
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
      y_extents * z_extents, 0.);
  Variables<tmpl::list<DataBoxTest_detail::vector>> expected_vars_sliced_in_y(
      x_extents * z_extents, 0.);
  Variables<tmpl::list<DataBoxTest_detail::vector>> expected_vars_sliced_in_z(
      x_extents * y_extents, 0.);
  const size_t x_offset = 1;
  const size_t y_offset = 2;
  const size_t z_offset = 1;

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
      // [data_on_slice]
      db::data_on_slice(box, extents, 0, x_offset,
                        tmpl::list<DataBoxTest_detail::vector>{})
      // [data_on_slice]
      == expected_vars_sliced_in_x);
  CHECK(db::data_on_slice(box, extents, 1, y_offset,
                          tmpl::list<DataBoxTest_detail::vector>{}) ==
        expected_vars_sliced_in_y);
  CHECK(db::data_on_slice(box, extents, 2, z_offset,
                          tmpl::list<DataBoxTest_detail::vector>{}) ==
        expected_vars_sliced_in_z);
}

void test_data_on_slice() noexcept {
  INFO("test data on slice");
  const size_t x_extents = 2;
  const size_t y_extents = 3;
  const size_t z_extents = 4;
  const size_t vec_size = DataBoxTest_detail::vector::type::size();
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
      y_extents * z_extents, 0.);
  Variables<tmpl::list<DataBoxTest_detail::vector>> expected_vars_sliced_in_y(
      x_extents * z_extents, 0.);
  Variables<tmpl::list<DataBoxTest_detail::vector>> expected_vars_sliced_in_z(
      x_extents * y_extents, 0.);
  const size_t x_offset = 1;
  const size_t y_offset = 2;
  const size_t z_offset = 1;

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
}  // namespace

namespace {
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

template <size_t N>
struct Parent : db::SimpleTag {
  using type = std::pair<Boxed<int>, Boxed<double>>;
};
template <size_t N>
struct ParentCompute : Parent<N>, db::ComputeTag {
  using base = Parent<N>;
  using return_type = std::pair<Boxed<int>, Boxed<double>>;
  static void function(
      const gsl::not_null<return_type*> result,
      const std::pair<Boxed<int>, Boxed<double>>& arg) noexcept {
    count++;
    *result = std::make_pair(
        Boxed<int>(std::make_shared<int>(*arg.first + 1)),
        Boxed<double>(std::make_shared<double>(*arg.second * 2.)));
  }
  using argument_tags = tmpl::list<Parent<N - 1>>;
  static int count;
};

template <size_t N>
int ParentCompute<N>::count = 0;

template <size_t N>
struct First : db::SimpleTag {
  using type = Boxed<int>;

  static constexpr size_t index = 0;
};
template <size_t N>
struct Second : db::SimpleTag {
  using type = Boxed<double>;

  static constexpr size_t index = 1;
};
}  // namespace

namespace db {
template <size_t N>
struct Subitems<Parent<N>> {
  using type = tmpl::list<First<N>, Second<N>>;
  using tag = Parent<N>;

  template <typename Subtag>
  static void create_item(
      const gsl::not_null<typename tag::type*> parent_value,
      const gsl::not_null<typename Subtag::type*> sub_value) noexcept {
    *sub_value = std::get<Subtag::index>(*parent_value);
  }
};

template <size_t N>
struct Subitems<ParentCompute<N>> {
  using type = tmpl::list<First<N>, Second<N>>;
  using tag = ParentCompute<N>;

  template <typename Subtag>
  static const typename Subtag::type& create_compute_item(
      const typename tag::type& parent_value) noexcept {
    return std::get<Subtag::index>(parent_value);
  }
};
}  // namespace db

namespace {

void test_subitems() noexcept {
  INFO("test subitems");
  auto box = db::create<db::AddSimpleTags<Parent<0>>,
                        db::AddComputeTags<ParentCompute<1>>>(
      std::make_pair(Boxed<int>(std::make_shared<int>(5)),
                     Boxed<double>(std::make_shared<double>(3.5))));

  TestHelpers::db::test_reference_tag<
      ::Tags::Subitem<First<1>, ParentCompute<1>>>("First");

  CHECK(*db::get<First<0>>(box) == 5);
  CHECK(*db::get<First<1>>(box) == 6);
  CHECK(*db::get<Second<0>>(box) == 3.5);
  CHECK(*db::get<Second<1>>(box) == 7);

  db::mutate<Second<0>>(
      make_not_null(&box),
      [](const gsl::not_null<Boxed<double>*> x) noexcept { **x = 12.; });

  CHECK(*db::get<First<0>>(box) == 5);
  CHECK(*db::get<First<1>>(box) == 6);
  CHECK(*db::get<Second<0>>(box) == 12.);
  CHECK(*db::get<Second<1>>(box) == 24.);

  {
    const auto copy_box = serialize_and_deserialize(box);
    CHECK(*db::get<First<0>>(copy_box) == 5);
    CHECK(*db::get<First<1>>(copy_box) == 6);
    CHECK(*db::get<Second<0>>(copy_box) == 12.);
    CHECK(*db::get<Second<1>>(copy_box) == 24.);
  }

  static_assert(
      std::is_same_v<
          decltype(box),
          decltype(db::create_from<db::RemoveTags<Parent<2>>>(
              db::create_from<db::RemoveTags<>, db::AddSimpleTags<Parent<2>>>(
                  std::move(box),
                  std::make_pair(
                      Boxed<int>(std::make_shared<int>(5)),
                      Boxed<double>(std::make_shared<double>(3.5))))))>,
      "Failed testing that adding and removing a simple subitem does "
      "not change the type of the DataBox");

  static_assert(
      std::is_same_v<decltype(box),
                     decltype(db::create_from<db::RemoveTags<ParentCompute<2>>>(
                         db::create_from<db::RemoveTags<>, db::AddSimpleTags<>,
                                         db::AddComputeTags<ParentCompute<2>>>(
                             std::move(box))))>,
      "Failed testing that adding and removing a compute subitem does "
      "not change the type of the DataBox");
}

namespace test_databox_tags {
struct Tag0Int : db::SimpleTag {
  using type = int;
};

template <typename ArgumentTag>
struct OverloadType : db::SimpleTag {
  using type = double;
};

// [overload_compute_tag_type]
template <typename ArgumentTag>
struct OverloadTypeCompute : OverloadType<ArgumentTag>, db::ComputeTag {
  using base = OverloadType<ArgumentTag>;
  using return_type = double;
  static constexpr void function(const gsl::not_null<double*> result,
                                 const int& a) noexcept {
    *result = 5 * a;
  }

  static constexpr void function(const gsl::not_null<double*> result,
                                 const double a) noexcept {
    *result = 3.2 * a;
  }
  using argument_tags = tmpl::list<ArgumentTag>;
};
// [overload_compute_tag_type]

template <typename ArgumentTag0, typename ArgumentTag1 = void>
struct OverloadNumberOfArgs : db::SimpleTag {
  using type = double;
};

// [overload_compute_tag_number_of_args]
template <typename ArgumentTag0, typename ArgumentTag1 = void>
struct OverloadNumberOfArgsCompute
    : OverloadNumberOfArgs<ArgumentTag0, ArgumentTag1>,
      db::ComputeTag {
  using base = OverloadNumberOfArgs<ArgumentTag0, ArgumentTag1>;
  using return_type = double;

  static constexpr void function(const gsl::not_null<double*> result,
                                 const double a) noexcept {
    *result = 3.2 * a;
  }

  static constexpr void function(const gsl::not_null<double*> result,
                                 const double a, const double b) noexcept {
    *result = a * b;
  }

  using argument_tags =
      tmpl::conditional_t<std::is_same_v<void, ArgumentTag1>,
                          tmpl::list<ArgumentTag0>,
                          tmpl::list<ArgumentTag0, ArgumentTag1>>;
};
// [overload_compute_tag_number_of_args]

template <typename ArgumentTag>
struct Template : db::SimpleTag {
  using type = typename ArgumentTag::type;
};

// [overload_compute_tag_template]
template <typename ArgumentTag>
struct TemplateCompute : Template<ArgumentTag>, db::ComputeTag {
  using base = Template<ArgumentTag>;
  using return_type = typename ArgumentTag::type;

  template <typename T>
  static constexpr void function(const gsl::not_null<T*> result,
                                 const T& a) noexcept {
    *result = 5 * a;
  }

  using argument_tags = tmpl::list<ArgumentTag>;
};
// [overload_compute_tag_template]
}  // namespace test_databox_tags

void test_overload_compute_tags() noexcept {
  INFO("testing overload compute tags.");
  auto box = db::create<
      db::AddSimpleTags<test_databox_tags::Tag0, test_databox_tags::Tag0Int>,
      db::AddComputeTags<
          test_databox_tags::OverloadTypeCompute<test_databox_tags::Tag0>,
          test_databox_tags::OverloadTypeCompute<test_databox_tags::Tag0Int>,
          test_databox_tags::OverloadNumberOfArgsCompute<
              test_databox_tags::Tag0>,
          test_databox_tags::OverloadNumberOfArgsCompute<
              test_databox_tags::Tag0,
              test_databox_tags::OverloadType<test_databox_tags::Tag0>>,
          test_databox_tags::TemplateCompute<test_databox_tags::Tag0>,
          test_databox_tags::TemplateCompute<test_databox_tags::Tag0Int>>>(8.4,
                                                                           -3);
  CHECK(db::get<test_databox_tags::OverloadType<test_databox_tags::Tag0>>(
            box) == 8.4 * 3.2);
  CHECK(db::get<test_databox_tags::OverloadType<test_databox_tags::Tag0Int>>(
            box) == -3 * 5);
  CHECK(
      db::get<test_databox_tags::OverloadNumberOfArgs<test_databox_tags::Tag0>>(
          box) == 3.2 * 8.4);
  CHECK(db::get<test_databox_tags::OverloadNumberOfArgs<
            test_databox_tags::Tag0,
            test_databox_tags::OverloadType<test_databox_tags::Tag0>>>(box) ==
        8.4 * 3.2 * 8.4);
  CHECK(db::get<test_databox_tags::Template<test_databox_tags::Tag0>>(box) ==
        8.4 * 5.0);
  CHECK(db::get<test_databox_tags::Template<test_databox_tags::Tag0Int>>(box) ==
        -3 * 5);
}

namespace TestTags {
namespace {
struct MyTag0 {
  using type = int;
};

struct MyTag1 {
  using type = double;
};

struct TupleTag : db::SimpleTag {
  using type = tuples::TaggedTuple<MyTag0, MyTag1>;
};
}  // namespace
}  // namespace TestTags

void test_with_tagged_tuple() noexcept {
  // Test that having a TaggedTuple inside a DataBox works properly
  auto box = db::create<db::AddSimpleTags<TestTags::TupleTag>>(
      tuples::TaggedTuple<TestTags::MyTag0, TestTags::MyTag1>{123, 2.3});
  auto box2 = std::move(box);
  CHECK(tuples::get<TestTags::MyTag0>(db::get<TestTags::TupleTag>(box2)) ==
        123);
  CHECK(tuples::get<TestTags::MyTag1>(db::get<TestTags::TupleTag>(box2)) ==
        2.3);
}

void serialization_non_subitem_simple_items() noexcept {
  INFO("serialization of a DataBox with non-Subitem simple items only");
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

void serialization_subitems_simple_items() noexcept {
  INFO("serialization of a DataBox with Subitem and non-Subitem simple items");
  auto serialization_test_box =
      db::create<db::AddSimpleTags<test_databox_tags::Tag0, Parent<0>,
                                   test_databox_tags::Tag1,
                                   test_databox_tags::Tag2, Parent<1>>>(
          3.14,
          std::make_pair(Boxed<int>(std::make_shared<int>(5)),
                         Boxed<double>(std::make_shared<double>(3.5))),
          std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s,
          std::make_pair(Boxed<int>(std::make_shared<int>(9)),
                         Boxed<double>(std::make_shared<double>(-4.5))));
  const double* before_0 =
      &db::get<test_databox_tags::Tag0>(serialization_test_box);
  const std::vector<double>* before_1 =
      &db::get<test_databox_tags::Tag1>(serialization_test_box);
  const std::string* before_2 =
      &db::get<test_databox_tags::Tag2>(serialization_test_box);
  const std::pair<Boxed<int>, Boxed<double>>* before_parent0 =
      &db::get<Parent<0>>(serialization_test_box);
  const Boxed<int>* before_parent0f =
      &db::get<First<0>>(serialization_test_box);
  const Boxed<double>* before_parent0s =
      &db::get<Second<0>>(serialization_test_box);
  const std::pair<Boxed<int>, Boxed<double>>* before_parent1 =
      &db::get<Parent<1>>(serialization_test_box);
  const Boxed<int>* before_parent1f =
      &db::get<First<1>>(serialization_test_box);
  const Boxed<double>* before_parent1s =
      &db::get<Second<1>>(serialization_test_box);

  auto deserialized_serialization_test_box =
      serialize_and_deserialize(serialization_test_box);
  CHECK(db::get<test_databox_tags::Tag0>(serialization_test_box) == 3.14);
  CHECK(db::get<test_databox_tags::Tag0>(deserialized_serialization_test_box) ==
        3.14);
  CHECK(before_0 == &db::get<test_databox_tags::Tag0>(serialization_test_box));
  CHECK(before_0 !=
        &db::get<test_databox_tags::Tag0>(deserialized_serialization_test_box));
  CHECK(*db::get<First<0>>(serialization_test_box) == 5);
  CHECK(*db::get<Second<0>>(serialization_test_box) == 3.5);
  CHECK(*db::get<First<0>>(deserialized_serialization_test_box) == 5);
  CHECK(*db::get<Second<0>>(deserialized_serialization_test_box) == 3.5);
  CHECK(before_parent0 == &db::get<Parent<0>>(serialization_test_box));
  CHECK(before_parent0 !=
        &db::get<Parent<0>>(deserialized_serialization_test_box));
  CHECK(before_parent0f == &db::get<First<0>>(serialization_test_box));
  CHECK(before_parent0f !=
        &db::get<First<0>>(deserialized_serialization_test_box));
  CHECK(before_parent0s == &db::get<Second<0>>(serialization_test_box));
  CHECK(before_parent0s !=
        &db::get<Second<0>>(deserialized_serialization_test_box));
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
  CHECK(*db::get<First<1>>(serialization_test_box) == 9);
  CHECK(*db::get<Second<1>>(serialization_test_box) == -4.5);
  CHECK(*db::get<First<1>>(deserialized_serialization_test_box) == 9);
  CHECK(*db::get<Second<1>>(deserialized_serialization_test_box) == -4.5);
  CHECK(before_parent1 == &db::get<Parent<1>>(serialization_test_box));
  CHECK(before_parent1 !=
        &db::get<Parent<1>>(deserialized_serialization_test_box));
  CHECK(before_parent1f == &db::get<First<1>>(serialization_test_box));
  CHECK(before_parent1f !=
        &db::get<First<1>>(deserialized_serialization_test_box));
  CHECK(before_parent1s == &db::get<Second<1>>(serialization_test_box));
  CHECK(before_parent1s !=
        &db::get<Second<1>>(deserialized_serialization_test_box));
}

template <int Id>
struct CountingFunc {
  static void apply(const gsl::not_null<double*> result) {
    count++;
    *result = 8.2;
  }
  static int count;
};

template <int Id>
int CountingFunc<Id>::count = 0;

template <int Id>
struct CountingTag : db::SimpleTag {
  using type = double;
};

template <int Id>
struct CountingTagCompute : CountingTag<Id>, db::ComputeTag {
  using base = CountingTag<Id>;
  using return_type = double;
  static constexpr auto function = CountingFunc<Id>::apply;
  using argument_tags = tmpl::list<>;
};

template <size_t SecondId>
struct CountingTagDouble : db::SimpleTag {
  using type = double;
};

template <size_t SecondId>
struct CountingTagDoubleCompute : CountingTagDouble<SecondId>, db::ComputeTag {
  using base = CountingTagDouble<SecondId>;
  using return_type = double;
  static void function(const gsl::not_null<double*> result,
                       const Boxed<double>& t) {
    count++;
    *result = *t * 6.0;
  }
  using argument_tags = tmpl::list<Second<SecondId>>;
  static int count;
};

template <size_t SecondId>
int CountingTagDoubleCompute<SecondId>::count = 0;

// clang-tidy: this function is too long. Yes, well we need to check lots
void serialization_subitem_compute_items() noexcept {  // NOLINT
  INFO("serialization of a DataBox with Subitem compute items");
  auto serialization_test_box =
      db::create<db::AddSimpleTags<test_databox_tags::Tag0, Parent<0>,
                                   test_databox_tags::Tag1,
                                   test_databox_tags::Tag2, Parent<1>>,
                 db::AddComputeTags<
                     CountingTagCompute<1>, test_databox_tags::Tag4Compute,
                     ParentCompute<2>, test_databox_tags::Tag5Compute,
                     ParentCompute<3>, CountingTagCompute<0>,
                     CountingTagDoubleCompute<2>, CountingTagDoubleCompute<3>>>(
          3.14,
          std::make_pair(Boxed<int>(std::make_shared<int>(5)),
                         Boxed<double>(std::make_shared<double>(3.5))),
          std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s,
          std::make_pair(Boxed<int>(std::make_shared<int>(9)),
                         Boxed<double>(std::make_shared<double>(-4.5))));
  const double* before_0 =
      &db::get<test_databox_tags::Tag0>(serialization_test_box);
  const std::vector<double>* before_1 =
      &db::get<test_databox_tags::Tag1>(serialization_test_box);
  const std::string* before_2 =
      &db::get<test_databox_tags::Tag2>(serialization_test_box);
  const std::pair<Boxed<int>, Boxed<double>>* before_parent0 =
      &db::get<Parent<0>>(serialization_test_box);
  const Boxed<int>* before_parent0f =
      &db::get<First<0>>(serialization_test_box);
  const Boxed<double>* before_parent0s =
      &db::get<Second<0>>(serialization_test_box);
  const std::pair<Boxed<int>, Boxed<double>>* before_parent1 =
      &db::get<Parent<1>>(serialization_test_box);
  const Boxed<int>* before_parent1f =
      &db::get<First<1>>(serialization_test_box);
  const Boxed<double>* before_parent1s =
      &db::get<Second<1>>(serialization_test_box);
  CHECK(db::get<test_databox_tags::Tag4>(serialization_test_box) == 6.28);
  const double* before_compute_tag0 =
      &db::get<test_databox_tags::Tag4>(serialization_test_box);
  CHECK(CountingFunc<0>::count == 0);
  CHECK(CountingFunc<1>::count == 0);
  CHECK(db::get<CountingTag<0>>(serialization_test_box) == 8.2);
  const double* before_counting_tag0 =
      &db::get<CountingTag<0>>(serialization_test_box);
  CHECK(CountingFunc<0>::count == 1);
  CHECK(CountingFunc<1>::count == 0);

  CHECK(ParentCompute<2>::count == 0);
  CHECK(ParentCompute<3>::count == 0);
  const std::pair<Boxed<int>, Boxed<double>>* before_parent2 =
      &db::get<Parent<2>>(serialization_test_box);
  CHECK(ParentCompute<2>::count == 1);
  CHECK(ParentCompute<3>::count == 0);
  const Boxed<int>* before_parent2_first =
      &db::get<First<2>>(serialization_test_box);
  const Boxed<double>* before_parent2_second =
      &db::get<Second<2>>(serialization_test_box);

  // Check we are correctly pointing into parent
  CHECK(&*(db::get<Parent<2>>(serialization_test_box).first) ==
        &*db::get<First<2>>(serialization_test_box));
  CHECK(&*(db::get<Parent<2>>(serialization_test_box).second) ==
        &*db::get<Second<2>>(serialization_test_box));

  CHECK(*(db::get<Parent<2>>(serialization_test_box).first) == 10);
  CHECK(*(db::get<Parent<2>>(serialization_test_box).second) == -9.0);

  CHECK(*db::get<First<2>>(serialization_test_box) == 10);
  CHECK(*db::get<Second<2>>(serialization_test_box) == -9.0);
  CHECK(ParentCompute<2>::count == 1);
  CHECK(ParentCompute<3>::count == 0);

  // Check compute items that take subitems
  CHECK(CountingTagDoubleCompute<2>::count == 0);
  CHECK(db::get<CountingTagDouble<2>>(serialization_test_box) == -9.0 * 6.0);
  CHECK(CountingTagDoubleCompute<2>::count == 1);
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
  CHECK(*db::get<First<0>>(serialization_test_box) == 5);
  CHECK(*db::get<Second<0>>(serialization_test_box) == 3.5);
  CHECK(*db::get<First<0>>(deserialized_serialization_test_box) == 5);
  CHECK(*db::get<Second<0>>(deserialized_serialization_test_box) == 3.5);
  CHECK(before_parent0 == &db::get<Parent<0>>(serialization_test_box));
  CHECK(before_parent0 !=
        &db::get<Parent<0>>(deserialized_serialization_test_box));
  CHECK(before_parent0f == &db::get<First<0>>(serialization_test_box));
  CHECK(before_parent0f !=
        &db::get<First<0>>(deserialized_serialization_test_box));
  CHECK(before_parent0s == &db::get<Second<0>>(serialization_test_box));
  CHECK(before_parent0s !=
        &db::get<Second<0>>(deserialized_serialization_test_box));
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
  CHECK(*db::get<First<1>>(serialization_test_box) == 9);
  CHECK(*db::get<Second<1>>(serialization_test_box) == -4.5);
  CHECK(*db::get<First<1>>(deserialized_serialization_test_box) == 9);
  CHECK(*db::get<Second<1>>(deserialized_serialization_test_box) == -4.5);
  CHECK(before_parent1 == &db::get<Parent<1>>(serialization_test_box));
  CHECK(before_parent1 !=
        &db::get<Parent<1>>(deserialized_serialization_test_box));
  CHECK(before_parent1f == &db::get<First<1>>(serialization_test_box));
  CHECK(before_parent1f !=
        &db::get<First<1>>(deserialized_serialization_test_box));
  CHECK(before_parent1s == &db::get<Second<1>>(serialization_test_box));
  CHECK(before_parent1s !=
        &db::get<Second<1>>(deserialized_serialization_test_box));
  // Check compute items
  CHECK(db::get<test_databox_tags::Tag4>(deserialized_serialization_test_box) ==
        6.28);
  CHECK(&db::get<test_databox_tags::Tag4>(
            deserialized_serialization_test_box) != before_compute_tag0);
  CHECK(db::get<test_databox_tags::Tag5>(deserialized_serialization_test_box) ==
        "My Sample String6.28"s);

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

  CHECK(ParentCompute<2>::count == 1);
  CHECK(ParentCompute<3>::count == 0);
  CHECK(&db::get<Parent<2>>(serialization_test_box) == before_parent2);
  // Check we are correctly pointing into parent
  CHECK(&*(db::get<Parent<2>>(serialization_test_box).first) ==
        &*db::get<First<2>>(serialization_test_box));
  CHECK(&*(db::get<Parent<2>>(serialization_test_box).second) ==
        &*db::get<Second<2>>(serialization_test_box));
  // Check that we did not reset the subitems items in the initial DataBox
  CHECK(&db::get<First<2>>(serialization_test_box) == before_parent2_first);
  CHECK(&db::get<Second<2>>(serialization_test_box) == before_parent2_second);
  CHECK(*(db::get<Parent<2>>(serialization_test_box).first) == 10);
  CHECK(*(db::get<Parent<2>>(serialization_test_box).second) == -9.0);
  CHECK(*(db::get<Parent<2>>(deserialized_serialization_test_box).first) == 10);
  CHECK(&db::get<Parent<2>>(deserialized_serialization_test_box) !=
        before_parent2);
  CHECK(*(db::get<Parent<2>>(deserialized_serialization_test_box).second) ==
        -9.0);
  CHECK(*db::get<First<2>>(deserialized_serialization_test_box) == 10);
  CHECK(*db::get<Second<2>>(deserialized_serialization_test_box) == -9.0);
  CHECK(ParentCompute<2>::count == 1);
  CHECK(ParentCompute<3>::count == 0);
  CHECK(&db::get<Parent<2>>(deserialized_serialization_test_box) !=
        before_parent2);
  // Check pointers in deserialized box
  CHECK(&db::get<First<2>>(deserialized_serialization_test_box) !=
        before_parent2_first);
  CHECK(&db::get<Second<2>>(deserialized_serialization_test_box) !=
        before_parent2_second);
  // Check we are correctly pointing into new parent and not old
  CHECK(&*(db::get<Parent<2>>(deserialized_serialization_test_box).first) ==
        &*db::get<First<2>>(deserialized_serialization_test_box));
  CHECK(&*(db::get<Parent<2>>(deserialized_serialization_test_box).second) ==
        &*db::get<Second<2>>(deserialized_serialization_test_box));
  CHECK(&*(db::get<Parent<2>>(deserialized_serialization_test_box).first) !=
        &*db::get<First<2>>(serialization_test_box));
  CHECK(&*(db::get<Parent<2>>(deserialized_serialization_test_box).second) !=
        &*db::get<Second<2>>(serialization_test_box));

  CHECK(*(db::get<Parent<3>>(serialization_test_box).first) == 11);
  CHECK(ParentCompute<2>::count == 1);
  CHECK(ParentCompute<3>::count == 1);
  CHECK(*(db::get<Parent<3>>(serialization_test_box).second) == -18.0);
  CHECK(ParentCompute<2>::count == 1);
  CHECK(ParentCompute<3>::count == 1);
  CHECK(*db::get<First<3>>(serialization_test_box) == 11);
  CHECK(*db::get<Second<3>>(serialization_test_box) == -18.0);
  CHECK(ParentCompute<2>::count == 1);
  CHECK(ParentCompute<3>::count == 1);
  CHECK(*(db::get<Parent<3>>(deserialized_serialization_test_box).first) == 11);
  CHECK(ParentCompute<2>::count == 1);
  CHECK(ParentCompute<3>::count == 2);
  CHECK(*(db::get<Parent<3>>(deserialized_serialization_test_box).second) ==
        -18.0);
  CHECK(*db::get<First<3>>(deserialized_serialization_test_box) == 11);
  CHECK(*db::get<Second<3>>(deserialized_serialization_test_box) == -18.0);
  CHECK(ParentCompute<2>::count == 1);
  CHECK(ParentCompute<3>::count == 2);

  // Check that all the Parent<3> related objects point to the right place
  CHECK(&*(db::get<Parent<3>>(deserialized_serialization_test_box).first) ==
        &*db::get<First<3>>(deserialized_serialization_test_box));
  CHECK(&*(db::get<Parent<3>>(deserialized_serialization_test_box).second) ==
        &*db::get<Second<3>>(deserialized_serialization_test_box));
  CHECK(&*(db::get<Parent<3>>(serialization_test_box).first) ==
        &*db::get<First<3>>(serialization_test_box));
  CHECK(&*(db::get<Parent<3>>(serialization_test_box).second) ==
        &*db::get<Second<3>>(serialization_test_box));
  CHECK(&*db::get<First<3>>(deserialized_serialization_test_box) !=
        &*db::get<First<3>>(serialization_test_box));
  CHECK(&*db::get<Second<3>>(deserialized_serialization_test_box) !=
        &*db::get<Second<3>>(serialization_test_box));

  // Check compute items that depend on the subitems
  CHECK(CountingTagDoubleCompute<2>::count == 1);
  CHECK(db::get<CountingTagDouble<2>>(serialization_test_box) == -9.0 * 6.0);
  CHECK(before_compute_tag2 ==
        &db::get<CountingTagDouble<2>>(serialization_test_box));
  CHECK(db::get<CountingTagDouble<2>>(deserialized_serialization_test_box) ==
        -9.0 * 6.0);
  CHECK(before_compute_tag2 !=
        &db::get<CountingTagDouble<2>>(deserialized_serialization_test_box));
  CHECK(CountingTagDoubleCompute<2>::count == 1);

  CHECK(CountingTagDoubleCompute<3>::count == 0);
  CHECK(db::get<CountingTagDouble<3>>(serialization_test_box) == -18.0 * 6.0);
  CHECK(db::get<CountingTagDouble<3>>(deserialized_serialization_test_box) ==
        -18.0 * 6.0);
  CHECK(&db::get<CountingTagDouble<3>>(serialization_test_box) !=
        &db::get<CountingTagDouble<3>>(deserialized_serialization_test_box));
  CHECK(CountingTagDoubleCompute<3>::count == 2);

  // Mutate subitems 1 in deserialized to see that changes propagate correctly
  db::mutate<Second<1>>(
      make_not_null(&serialization_test_box),
      [](const gsl::not_null<Boxed<double>*> x) noexcept { **x = 12.; });
  CHECK(ParentCompute<2>::count == 1);
  CHECK(CountingTagDoubleCompute<2>::count == 1);
  CHECK(db::get<CountingTagDouble<2>>(serialization_test_box) == 24.0 * 6.0);
  CHECK(ParentCompute<2>::count == 2);
  CHECK(CountingTagDoubleCompute<2>::count == 2);
  CHECK(CountingTagDoubleCompute<3>::count == 2);
  CHECK(db::get<CountingTagDouble<3>>(serialization_test_box) == 48.0 * 6.0);
  CHECK(CountingTagDoubleCompute<3>::count == 3);

  db::mutate<Second<1>>(
      make_not_null(&deserialized_serialization_test_box),
      [](const gsl::not_null<Boxed<double>*> x) noexcept { **x = -7.; });
  CHECK(ParentCompute<2>::count == 2);
  CHECK(CountingTagDoubleCompute<2>::count == 2);
  CHECK(db::get<CountingTagDouble<2>>(deserialized_serialization_test_box) ==
        -14.0 * 6.0);
  CHECK(ParentCompute<2>::count == 3);
  CHECK(CountingTagDoubleCompute<2>::count == 3);
  CHECK(CountingTagDoubleCompute<3>::count == 3);
  CHECK(db::get<CountingTagDouble<3>>(deserialized_serialization_test_box) ==
        -28.0 * 6.0);
  CHECK(CountingTagDoubleCompute<3>::count == 4);

  // Check things didn't get modified in the original DataBox
  CHECK(ParentCompute<2>::count == 3);
  CHECK(CountingTagDoubleCompute<2>::count == 3);
  CHECK(db::get<CountingTagDouble<2>>(serialization_test_box) == 24.0 * 6.0);
  CHECK(ParentCompute<2>::count == 3);
  CHECK(CountingTagDoubleCompute<2>::count == 3);
  CHECK(CountingTagDoubleCompute<3>::count == 4);
  CHECK(db::get<CountingTagDouble<3>>(serialization_test_box) == 48.0 * 6.0);
  CHECK(CountingTagDoubleCompute<3>::count == 4);

  CountingFunc<0>::count = 0;
  CountingFunc<1>::count = 0;
  CountingTagDoubleCompute<2>::count = 0;
  CountingTagDoubleCompute<3>::count = 0;
  ParentCompute<2>::count = 0;
  ParentCompute<3>::count = 0;
}

void serialization_compute_items_of_base_tags() noexcept {
  INFO("serialization of a DataBox with compute items depending on base tags");
  auto original_box =
      db::create<db::AddSimpleTags<test_databox_tags::Tag2>,
                 db::AddComputeTags<test_databox_tags::Tag6Compute>>(
          "My Sample String"s);
  CHECK(db::get<test_databox_tags::Tag2>(original_box) == "My Sample String");
  CHECK(db::get<test_databox_tags::Tag6>(original_box) == "My Sample String");
  auto copied_box = serialize_and_deserialize(original_box);
  CHECK(db::get<test_databox_tags::Tag2>(copied_box) == "My Sample String");
  CHECK(db::get<test_databox_tags::Tag6>(copied_box) == "My Sample String");
}

void serialization_of_pointers() noexcept {
  INFO("Serialization of pointers");
  const auto box =
      db::create<db::AddSimpleTags<test_databox_tags::Pointer>,
                 db::AddComputeTags<test_databox_tags::PointerToCounterCompute,
                                    test_databox_tags::PointerToSumCompute>>(
          std::make_unique<int>(3));
  const auto check = [](const decltype(box)& check_box) noexcept {
    CHECK(db::get<test_databox_tags::Pointer>(check_box) == 3);
    CHECK(db::get<test_databox_tags::PointerToCounter>(check_box) == 4);
    CHECK(db::get<test_databox_tags::PointerToSum>(check_box) == 8);
  };
  check(serialize_and_deserialize(box));  // before compute items evaluated
  check(box);
  check(serialize_and_deserialize(box));  // after compute items evaluated
}

namespace test_databox_tags {
// [databox_reference_tag_example]
template <typename... Tags>
struct TaggedTuple : db::SimpleTag {
  using type = tuples::TaggedTuple<Tags...>;
};

template <typename Tag, typename ParentTag>
struct FromTaggedTuple : Tag, db::ReferenceTag {
  using base = Tag;
  using parent_tag = ParentTag;

  static const auto& get(const typename parent_tag::type& tagged_tuple) {
    return tuples::get<Tag>(tagged_tuple);
  }

  using argument_tags = tmpl::list<parent_tag>;
};
// [databox_reference_tag_example]
}  // namespace test_databox_tags

void test_reference_item() noexcept {
  INFO("test reference item");
  using tuple_tag = test_databox_tags::TaggedTuple<test_databox_tags::Tag0,
                                                   test_databox_tags::Tag1,
                                                   test_databox_tags::Tag2>;
  auto box =
      db::create<db::AddSimpleTags<tuple_tag>,
                 db::AddComputeTags<test_databox_tags::FromTaggedTuple<
                                        test_databox_tags::Tag0, tuple_tag>,
                                    test_databox_tags::FromTaggedTuple<
                                        test_databox_tags::Tag1, tuple_tag>,
                                    test_databox_tags::FromTaggedTuple<
                                        test_databox_tags::Tag2, tuple_tag>>>(
          tuples::TaggedTuple<test_databox_tags::Tag0, test_databox_tags::Tag1,
                              test_databox_tags::Tag2>{
              3.14, std::vector<double>{8.7, 93.2, 84.7}, "My Sample String"s});
  const auto& tagged_tuple = get<tuple_tag>(box);
  CHECK(get<test_databox_tags::Tag0>(tagged_tuple) == 3.14);
  CHECK(get<test_databox_tags::Tag1>(tagged_tuple) ==
        std::vector<double>{8.7, 93.2, 84.7});
  CHECK(get<test_databox_tags::Tag2>(tagged_tuple) == "My Sample String"s);
  CHECK(get<test_databox_tags::Tag0>(box) == 3.14);
  CHECK(get<test_databox_tags::Tag1>(box) ==
        std::vector<double>{8.7, 93.2, 84.7});
  CHECK(get<test_databox_tags::Tag2>(box) == "My Sample String"s);
}

void test_serialization() noexcept {
  serialization_non_subitem_simple_items();
  serialization_subitems_simple_items();
  serialization_subitem_compute_items();
  serialization_compute_items_of_base_tags();
  serialization_of_pointers();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox", "[Unit][DataStructures]") {
  test_databox();
  test_create_argument_types();
  test_get_databox();
  test_mutate();
  test_apply();
  test_variables();
  test_variables2();
  test_reset_compute_items();
  test_variables_extra_reset();
  test_mutate_apply();
  test_mutating_compute_item();
  test_data_on_slice_single();
  test_data_on_slice();
  test_subitems();
  test_overload_compute_tags();
  test_with_tagged_tuple();
  test_serialization();
  test_reference_item();
}

// Test`tag_is_retrievable_v`
namespace {
namespace tags_types {
struct PureBaseTag : db::BaseTag {};

struct SimpleTag : PureBaseTag, db::SimpleTag {
  using type = double;
};

struct DummyTag : db::SimpleTag {
  using type = int;
};
}  // namespace tags_types

static_assert(
    db::tag_is_retrievable_v<tags_types::PureBaseTag,
                             db::DataBox<tmpl::list<tags_types::SimpleTag>>>,
    "Failed testing tag_is_retrievable_v");
static_assert(
    db::tag_is_retrievable_v<tags_types::SimpleTag,
                             db::DataBox<tmpl::list<tags_types::SimpleTag>>>,
    "Failed testing tag_is_retrievable_v");
static_assert(
    not db::tag_is_retrievable_v<
        tags_types::DummyTag, db::DataBox<tmpl::list<tags_types::SimpleTag>>>,
    "Failed testing tag_is_retrievable_v");
}  // namespace
