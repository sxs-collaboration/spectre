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
template <typename T>
void test_instance(Options::Auto<T> value,
                   const std::optional<T> expected) noexcept {
  CHECK(static_cast<const std::optional<T>&>(value) == expected);
  CHECK(static_cast<std::optional<T>>(std::move(value)) == expected);
}

void test_class() noexcept {
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

  CHECK(get_output(Options::Auto<int>{}) == "Auto");
  CHECK(get_output(Options::Auto<int>{3}) == "3");
  CHECK(get_output(Options::Auto<std::vector<int>>{{1, 2}}) ==
        get_output(std::vector<int>{1, 2}));
}

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
void check_create(const std::string& creation_string,
                  const std::optional<T>& expected) noexcept {
  CAPTURE(creation_string);
  CHECK(static_cast<std::optional<T>>(
            TestHelpers::test_creation<Options::Auto<T>>(creation_string)) ==
        expected);
}

void test_parsing() noexcept {
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
/// [example_class]
class ExampleClass {
 public:
  ExampleClass() = default;

  struct AutoArg {
    using type = Options::Auto<int>;
    static type default_value() noexcept { return {}; }
    static constexpr Options::String help =
        "Integer that can be automatically chosen";
  };

  static constexpr Options::String help =
      "A class that can automatically choose an argument";
  using options = tmpl::list<AutoArg>;

  explicit ExampleClass(std::optional<int> auto_arg) noexcept
      : value(auto_arg ? *auto_arg : -12) {}

  int value{};
};
/// [example_class]
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
      std::optional<std::unique_ptr<Base>> /*auto_arg*/) noexcept {}
};

void test_use_as_option() noexcept {
  /// [example_create]
  CHECK(TestHelpers::test_creation<ExampleClass>("AutoArg: 7").value == 7);
  CHECK(TestHelpers::test_creation<ExampleClass>("AutoArg: Auto").value == -12);
  /// [example_create]

  // Make sure this compiles.
  TestHelpers::test_creation<NonCopyableArgument>("AutoArg: Auto");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Options.Auto", "[Unit][Options]") {
  test_class();
  test_parsing();
  test_use_as_option();
}
