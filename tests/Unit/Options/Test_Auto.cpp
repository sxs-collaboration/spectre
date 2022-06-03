// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Framework/TestCreation.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Derived;

struct Base {
  Base() = default;
  Base(const Base&) = default;
  Base& operator=(Base&&) = default;
  Base(Base&&) = default;
  Base& operator=(const Base&) = default;
  virtual ~Base() = default;
  using creatable_classes = tmpl::list<Derived>;
};

struct Derived : Base {
  using options = tmpl::list<>;
  static constexpr Options::String help = "halp";
};

template <typename T>
void test_instance(Options::Auto<T> value, const std::optional<T> expected) {
  CHECK(static_cast<const std::optional<T>&>(value) == expected);
  CHECK(static_cast<std::optional<T>>(std::move(value)) == expected);
}

void test_class() {
  test_instance(Options::Auto<int>{}, std::optional<int>{});
  test_instance(Options::Auto<int>{3}, std::optional<int>{3});
  test_instance(Options::Auto<std::unique_ptr<int>>{},
                std::optional<std::unique_ptr<int>>{});
  {
    Options::Auto<std::unique_ptr<int>> value{std::make_unique<int>(3)};
    const std::optional<std::unique_ptr<int>>& contained = value;
    CHECK((contained and *contained and **contained == 3));
    const std::optional<std::unique_ptr<int>> extracted = std::move(value);
    CHECK((extracted and *extracted and **extracted == 3));
  }

  {
    struct A {};
    struct B {
      // NOLINTNEXTLINE(google-explicit-constructor)
      B(A /*unused*/) {}
    };

    // Test that converting the contained class compiles.
    (void)static_cast<std::optional<B>>(Options::Auto<A>{});

    // Similarly, but for the specific case we care about most.
    (void)static_cast<std::optional<std::unique_ptr<Base>>>(
        Options::Auto<std::unique_ptr<Derived>>{});
  }

  CHECK(Options::Auto<int>{} == Options::Auto<int>{});
  CHECK_FALSE(Options::Auto<int>{} != Options::Auto<int>{});
  CHECK(Options::Auto<int>{3} == Options::Auto<int>{3});
  CHECK_FALSE(Options::Auto<int>{3} != Options::Auto<int>{3});
  CHECK_FALSE(Options::Auto<int>{} == Options::Auto<int>{3});
  CHECK(Options::Auto<int>{} != Options::Auto<int>{3});
  CHECK_FALSE(Options::Auto<int>{3} == Options::Auto<int>{4});
  CHECK(Options::Auto<int>{3} != Options::Auto<int>{4});

  CHECK(get_output(Options::Auto<int>{}) == "Auto");
  CHECK(get_output(Options::Auto<int>{3}) == "3");
  CHECK(get_output(Options::Auto<std::vector<int>>{{1, 2}}) ==
        get_output(std::vector<int>{1, 2}));
}

template <typename T>
void check_create(const std::string& creation_string,
                  const std::optional<T>& expected) {
  CAPTURE(creation_string);
  CHECK(static_cast<std::optional<T>>(
            TestHelpers::test_creation<Options::Auto<T>>(creation_string)) ==
        expected);
}

void test_parsing() {
  check_create<int>("3", 3);
  check_create<int>("Auto", {});

  check_create<std::string>("Auto", {});
  check_create<std::string>("Not Auto", "Not Auto");

  check_create<std::unique_ptr<Base>>("Auto", {});
  {
    const std::optional<std::unique_ptr<Base>> created =
        TestHelpers::test_creation<Options::Auto<std::unique_ptr<Base>>>(
            "Derived");
    CHECK(created);
    CHECK(*created);
  }

  check_create<std::vector<int>>("Auto", {});
  check_create<std::vector<int>>("", std::vector<int>{});
  check_create<std::vector<int>>("[1, 2, 3]", std::vector<int>{1, 2, 3});
}

#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 8 && __GNUC__ < 11
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif  // defined(__GNUC__) && !defined(__clang__) && __GNUC__ => 8 && __GNUC__ < 11
// [example_class]
class ExampleClass {
 public:
  ExampleClass() = default;

  struct AutoArg {
    using type = Options::Auto<int>;
    static type suggested_value() { return {}; }
    static constexpr Options::String help =
        "Integer that can be automatically chosen";
  };
  struct OptionalArg {
    using type = Options::Auto<double, Options::AutoLabel::None>;
    static constexpr Options::String help = "Optional parameter";
  };
  struct AllArg {
    using type = Options::Auto<std::vector<int>, Options::AutoLabel::All>;
    static constexpr Options::String help = "Optional parameter all";
  };

  static constexpr Options::String help =
      "A class that can automatically choose an argument";
  using options = tmpl::list<AutoArg, OptionalArg, AllArg>;

  explicit ExampleClass(std::optional<int> auto_arg,
                        std::optional<double> opt_arg,
                        std::optional<std::vector<int>> all_arg)
      : value(auto_arg ? *auto_arg : -12),
        optional_value(opt_arg),
        all_value(all_arg) {}

  int value{};
  std::optional<double> optional_value{};
  std::optional<std::vector<int>> all_value{};
};
// [example_class]
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 8 && __GNUC__ < 11
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && !defined(__clang__) && __GNUC__ => 8 && __GNUC__ < 11

class NonCopyableArgument {
 public:
  NonCopyableArgument() = default;

  struct AutoArg {
    using type = Options::Auto<std::unique_ptr<Base>>;
    static constexpr Options::String help = "halp";
  };

  static constexpr Options::String help = "halp";
  using options = tmpl::list<AutoArg>;

  explicit NonCopyableArgument(
      std::optional<std::unique_ptr<Base>> /*auto_arg*/) {}
};

void test_use_as_option() {
  // [example_create]
  const auto example1 = TestHelpers::test_creation<ExampleClass>(
      "AutoArg: 7\n"
      "OptionalArg: 10.\n"
      "AllArg: [0, 1, 2]");
  CHECK(example1.value == 7);
  CHECK(example1.optional_value == 10.);
  CHECK(example1.all_value == std::vector<int>{{0, 1, 2}});
  const auto example2 = TestHelpers::test_creation<ExampleClass>(
      "AutoArg: Auto\n"
      "OptionalArg: None\n"
      "AllArg: [0, 1, 2]");
  CHECK(example2.value == -12);
  CHECK(example2.optional_value == std::nullopt);
  CHECK(example2.all_value == std::vector<int>{{0, 1, 2}});
  const auto example3 = TestHelpers::test_creation<ExampleClass>(
      "AutoArg: 7\n"
      "OptionalArg: 10.\n"
      "AllArg: All");
  CHECK(example3.value == 7);
  CHECK(example3.optional_value == 10.);
  CHECK(example3.all_value == std::nullopt);
  // [example_create]

  // Make sure this compiles.
  TestHelpers::test_creation<NonCopyableArgument>("AutoArg: Auto");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Options.Auto", "[Unit][Options]") {
  test_class();
  test_parsing();
  test_use_as_option();
}
