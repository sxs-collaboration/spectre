// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <ostream>
#include <string>

#include "Framework/TestCreation.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/TMPL.hpp"

namespace {
enum class Color { Red, Green, Purple };
}  // namespace

template <>
struct Options::create_from_yaml<Color> {
  template <typename Metavariables>
  static Color create(const Options::Option& options) {
    return create<void>(options);
  }
};
template <>
Color Options::create_from_yaml<Color>::create<void>(
    const Options::Option& options) {
  const auto color_read = options.parse_as<std::string>();
  if (color_read == "Red") {
    return Color::Red;
  } else if (color_read == "Green") {
    return Color::Green;
  } else if (color_read == "Purple") {
    return Color::Purple;
  }
  PARSE_ERROR(options.context(), "Failed to convert \""
                                     << color_read
                                     << "\" to Color. Expected one of: "
                                        "{Red, Green, Purple}.");
}

namespace {
// [class_without_metavariables]
struct ClassWithoutMetavariables {
  struct SizeT {
    using type = size_t;
    static constexpr Options::String help = {"SizeT help"};
  };

  using options = tmpl::list<SizeT>;
  static constexpr Options::String help = {"Help"};

  explicit ClassWithoutMetavariables(const size_t in_value) : value(in_value) {}

  ClassWithoutMetavariables() = default;

  size_t value{0};
};
// [class_without_metavariables]

struct ClassWithMetavariables {
  struct SizeT {
    using type = size_t;
    static constexpr Options::String help = {"SizeT help"};
  };

  using options = tmpl::list<SizeT>;
  static constexpr Options::String help = {"Help"};

  template <typename Metavariables>
  // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
  ClassWithMetavariables(const size_t in_value,
                         const Options::Context& /*context*/,
                         Metavariables /*meta*/) {
    value = in_value * Metavariables::value_multiplier;
  }

  ClassWithMetavariables(const size_t /*in_value*/,
                         const Options::Context& /*context*/,
                         NoSuchType /*meta*/)
      : value{std::numeric_limits<size_t>::max()} {}

  ClassWithMetavariables() = default;

  size_t value{0};
};

struct OptionGroup1 {
  static constexpr Options::String help = {"OptionGroup1 help"};
};

struct OptionGroup2 {
  static constexpr Options::String help = {"OptionGroup2 help"};
  using group = OptionGroup1;
};

template <typename T>
struct NoGroup {
  using type = T;
  static constexpr Options::String help = {"halp"};
};

template <typename T>
struct OneGroup {
  using type = T;
  static constexpr Options::String help = {"halp"};
  using group = OptionGroup1;
};

template <typename T>
struct TwoGroup {
  using type = T;
  static constexpr Options::String help = {"halp"};
  using group = OptionGroup2;
};

template <size_t ValueMultiplier>
struct Metavars {
  static constexpr size_t value_multiplier = ValueMultiplier;
};

// [class_without_metavariables_tag]
struct ExampleTag {
  using type = ClassWithoutMetavariables;
  static constexpr Options::String help = {"help"};
  using group = OptionGroup1;
};
// [class_without_metavariables_tag]

struct BaseClass {
  BaseClass() = default;
  BaseClass(const BaseClass&) = delete;
  BaseClass& operator=(const BaseClass&) = delete;
  BaseClass(BaseClass&&) = default;
  BaseClass& operator=(BaseClass&&) = default;
  virtual ~BaseClass() = default;

  virtual size_t get_value() const = 0;
};

struct DerivedClass : BaseClass {
  struct SizeT {
    using type = size_t;
    static constexpr Options::String help = {"SizeT help"};
  };

  using options = tmpl::list<SizeT>;
  static constexpr Options::String help = {"Help"};

  explicit DerivedClass(const size_t in_value) : value(in_value) {}

  DerivedClass() = default;
  DerivedClass(const DerivedClass&) = delete;
  DerivedClass& operator=(const DerivedClass&) = delete;
  DerivedClass(DerivedClass&&) = default;
  DerivedClass& operator=(DerivedClass&&) = default;
  ~DerivedClass() override = default;

  size_t get_value() const override { return value; }

  size_t value{0};
};

void test_test_creation() {
  // Test creation of fundamentals
  CHECK(TestHelpers::test_creation<double>("1.846") == 1.846);
  CHECK(TestHelpers::test_option_tag<NoGroup<double>>("1.846") == 1.846);
  CHECK(TestHelpers::test_option_tag<OneGroup<double>>("1.846") == 1.846);
  CHECK(TestHelpers::test_option_tag<TwoGroup<double>>("1.846") == 1.846);

  // Test class that doesn't need metavariables when not passing metavariables
  // [class_without_metavariables_create]
  CHECK(
      TestHelpers::test_creation<ClassWithoutMetavariables>("SizeT: 7").value ==
      7);
  // [class_without_metavariables_create]
  // [class_without_metavariables_create_tag]
  CHECK(TestHelpers::test_option_tag<ExampleTag>("SizeT: 7").value == 7);
  // [class_without_metavariables_create_tag]

  CHECK(TestHelpers::test_option_tag<NoGroup<ClassWithoutMetavariables>>(
            "SizeT: 4")
            .value == 4);
  CHECK(TestHelpers::test_option_tag<OneGroup<ClassWithoutMetavariables>>(
            "SizeT: 5")
            .value == 5);
  CHECK(TestHelpers::test_option_tag<TwoGroup<ClassWithoutMetavariables>>(
            "SizeT: 6")
            .value == 6);

  // Test class that doesn't need metavariables but passing metavariables
  CHECK(TestHelpers::test_creation<ClassWithoutMetavariables, Metavars<3>>(
            "SizeT: 8")
            .value == 8);
  CHECK(TestHelpers::test_option_tag<NoGroup<ClassWithoutMetavariables>,
                                     Metavars<4>>("SizeT: 9")
            .value == 9);
  CHECK(TestHelpers::test_option_tag<OneGroup<ClassWithoutMetavariables>,
                                     Metavars<5>>("SizeT: 10")
            .value == 10);
  CHECK(TestHelpers::test_option_tag<TwoGroup<ClassWithoutMetavariables>,
                                     Metavars<6>>("SizeT: 11")
            .value == 11);

  // Test class that uses metavariables but not passing metavariables
  CHECK(
      TestHelpers::test_option_tag<NoGroup<ClassWithMetavariables>>("SizeT: 4")
          .value == std::numeric_limits<size_t>::max());
  CHECK(
      TestHelpers::test_option_tag<OneGroup<ClassWithMetavariables>>("SizeT: 4")
          .value == std::numeric_limits<size_t>::max());
  CHECK(
      TestHelpers::test_option_tag<TwoGroup<ClassWithMetavariables>>("SizeT: 4")
          .value == std::numeric_limits<size_t>::max());

  // Test class that uses metavariables but passing metavariables
  CHECK(TestHelpers::test_creation<ClassWithMetavariables, Metavars<3>>(
            "SizeT: 4")
            .value == 12);
  CHECK(TestHelpers::test_option_tag<NoGroup<ClassWithMetavariables>,
                                     Metavars<4>>("SizeT: 4")
            .value == 16);
  CHECK(TestHelpers::test_option_tag<OneGroup<ClassWithMetavariables>,
                                     Metavars<5>>("SizeT: 4")
            .value == 20);
  CHECK(TestHelpers::test_option_tag<TwoGroup<ClassWithMetavariables>,
                                     Metavars<6>>("SizeT: 4")
            .value == 24);
}

void test_test_factory_creation() {
  // [test_factory_creation]
  CHECK(TestHelpers::test_factory_creation<BaseClass, DerivedClass>(
            "DerivedClass:\n"
            "  SizeT: 5")
            ->get_value() == 5);
  // [test_factory_creation]
}

void test_test_enum_creation() {
  CHECK(TestHelpers::test_creation<Color>("Purple") == Color::Purple);
  CHECK(TestHelpers::test_option_tag<NoGroup<Color>>("Purple") ==
        Color::Purple);
  CHECK(TestHelpers::test_option_tag<OneGroup<Color>>("Purple") ==
        Color::Purple);
  CHECK(TestHelpers::test_option_tag<TwoGroup<Color>>("Purple") ==
        Color::Purple);
  CHECK(TestHelpers::test_creation<Color, Metavars<3>>("Purple") ==
        Color::Purple);
  CHECK(TestHelpers::test_option_tag<NoGroup<Color>, Metavars<3>>("Purple") ==
        Color::Purple);
  CHECK(TestHelpers::test_option_tag<OneGroup<Color>, Metavars<3>>("Purple") ==
        Color::Purple);
  CHECK(TestHelpers::test_option_tag<TwoGroup<Color>, Metavars<3>>("Purple") ==
        Color::Purple);
}

SPECTRE_TEST_CASE("Unit.TestCreation", "[Unit]") {
  test_test_creation();
  test_test_factory_creation();
  test_test_enum_creation();
}
}  // namespace
