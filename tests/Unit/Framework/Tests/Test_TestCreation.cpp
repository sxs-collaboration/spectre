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

struct DerivedClassWithoutMetavariables;
struct DerivedClassWithMetavariables;

struct BaseClass {
  using creatable_classes = tmpl::list<DerivedClassWithoutMetavariables,
                                       DerivedClassWithMetavariables>;
  BaseClass() = default;
  BaseClass(const BaseClass&) = delete;
  BaseClass& operator=(const BaseClass&) = delete;
  BaseClass(BaseClass&&) = default;
  BaseClass& operator=(BaseClass&&) = default;
  virtual ~BaseClass() = default;

  virtual size_t get_value() const = 0;
};

struct DerivedClassWithoutMetavariables : BaseClass {
  struct SizeT {
    using type = size_t;
    static constexpr Options::String help = {"SizeT help"};
  };

  using options = tmpl::list<SizeT>;
  static constexpr Options::String help = {"Help"};

  explicit DerivedClassWithoutMetavariables(const size_t in_value)
      : value(in_value) {}

  DerivedClassWithoutMetavariables() = default;
  DerivedClassWithoutMetavariables(const DerivedClassWithoutMetavariables&) =
      delete;
  DerivedClassWithoutMetavariables& operator=(
      const DerivedClassWithoutMetavariables&) = delete;
  DerivedClassWithoutMetavariables(DerivedClassWithoutMetavariables&&) =
      default;
  DerivedClassWithoutMetavariables& operator=(
      DerivedClassWithoutMetavariables&&) = default;
  ~DerivedClassWithoutMetavariables() override = default;

  size_t get_value() const override { return value; }

  size_t value{0};
};

struct DerivedClassWithMetavariables : BaseClass {
  struct SizeT {
    using type = size_t;
    static constexpr Options::String help = {"SizeT help"};
  };

  using options = tmpl::list<SizeT>;
  static constexpr Options::String help = {"Help"};

  template <typename Metavariables>
  // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
  DerivedClassWithMetavariables(const size_t in_value,
                                const Options::Context& /*context*/,
                                Metavariables /*meta*/) {
    value = in_value * Metavariables::value_multiplier;
  }

  DerivedClassWithMetavariables(const size_t /*in_value*/,
                                const Options::Context& /*context*/,
                                NoSuchType /*meta*/)
      : value{std::numeric_limits<size_t>::max()} {}

  DerivedClassWithMetavariables() = default;
  DerivedClassWithMetavariables(const DerivedClassWithMetavariables&) = delete;
  DerivedClassWithMetavariables& operator=(
      const DerivedClassWithMetavariables&) = delete;
  DerivedClassWithMetavariables(DerivedClassWithMetavariables&&) = default;
  DerivedClassWithMetavariables& operator=(DerivedClassWithMetavariables&&) =
      default;
  ~DerivedClassWithMetavariables() override = default;

  size_t get_value() const override { return value; }

  size_t value{0};
};

void test_test_creation() {
  // Test creation of fundamentals
  CHECK(TestHelpers::test_creation<double>("1.846") == 1.846);
  CHECK(
      TestHelpers::test_creation<double, TestHelpers::TestCreationOpt<double>>(
          "1.846") == 1.846);
  CHECK(TestHelpers::test_creation<double, NoGroup<double>>("1.846") == 1.846);
  CHECK(TestHelpers::test_creation<double, OneGroup<double>>("1.846") == 1.846);
  CHECK(TestHelpers::test_creation<double, TwoGroup<double>>("1.846") == 1.846);

  // Test class that doesn't need metavariables when not passing metavariables
  // [size_t_argument]
  CHECK(
      TestHelpers::test_creation<ClassWithoutMetavariables>("SizeT: 7").value ==
      7);
  // [size_t_argument]

  CHECK(
      TestHelpers::test_creation<ClassWithoutMetavariables,
                                 NoGroup<ClassWithoutMetavariables>>("SizeT: 4")
          .value == 4);
  CHECK(TestHelpers::test_creation<ClassWithoutMetavariables,
                                   OneGroup<ClassWithoutMetavariables>>(
            "SizeT: 5")
            .value == 5);
  CHECK(TestHelpers::test_creation<ClassWithoutMetavariables,
                                   TwoGroup<ClassWithoutMetavariables>>(
            "SizeT: 6")
            .value == 6);

  // Test class that doesn't need metavariables but passing metavariables
  CHECK(
      TestHelpers::test_creation<
          ClassWithoutMetavariables,
          TestHelpers::TestCreationOpt<ClassWithoutMetavariables>, Metavars<3>>(
          "SizeT: 8")
          .value == 8);
  CHECK(TestHelpers::test_creation<ClassWithoutMetavariables,
                                   NoGroup<ClassWithoutMetavariables>,
                                   Metavars<4>>("SizeT: 9")
            .value == 9);
  CHECK(TestHelpers::test_creation<ClassWithoutMetavariables,
                                   OneGroup<ClassWithoutMetavariables>,
                                   Metavars<5>>("SizeT: 10")
            .value == 10);
  CHECK(TestHelpers::test_creation<ClassWithoutMetavariables,
                                   TwoGroup<ClassWithoutMetavariables>,
                                   Metavars<6>>("SizeT: 11")
            .value == 11);

  // Test class that uses metavariables but not passing metavariables
  CHECK(TestHelpers::test_creation<
            ClassWithMetavariables,
            TestHelpers::TestCreationOpt<ClassWithMetavariables>>("SizeT: 4")
            .value == std::numeric_limits<size_t>::max());
  CHECK(TestHelpers::test_creation<ClassWithMetavariables,
                                   NoGroup<ClassWithMetavariables>>("SizeT: 4")
            .value == std::numeric_limits<size_t>::max());
  CHECK(TestHelpers::test_creation<ClassWithMetavariables,
                                   OneGroup<ClassWithMetavariables>>("SizeT: 4")
            .value == std::numeric_limits<size_t>::max());
  CHECK(TestHelpers::test_creation<ClassWithMetavariables,
                                   TwoGroup<ClassWithMetavariables>>("SizeT: 4")
            .value == std::numeric_limits<size_t>::max());

  // Test class that uses metavariables but passing metavariables
  CHECK(TestHelpers::test_creation<
            ClassWithMetavariables,
            TestHelpers::TestCreationOpt<ClassWithMetavariables>, Metavars<3>>(
            "SizeT: 4")
            .value == 12);
  CHECK(
      TestHelpers::test_creation<ClassWithMetavariables,
                                 NoGroup<ClassWithMetavariables>, Metavars<4>>(
          "SizeT: 4")
          .value == 16);
  CHECK(
      TestHelpers::test_creation<ClassWithMetavariables,
                                 OneGroup<ClassWithMetavariables>, Metavars<5>>(
          "SizeT: 4")
          .value == 20);
  CHECK(
      TestHelpers::test_creation<ClassWithMetavariables,
                                 TwoGroup<ClassWithMetavariables>, Metavars<6>>(
          "SizeT: 4")
          .value == 24);
}

void test_test_factory_creation() {
  // Test derived class that doesn't need metavariables when not passing
  // metavariables
  // [size_t_argument_base]
  CHECK(TestHelpers::test_factory_creation<BaseClass>(
            "DerivedClassWithoutMetavariables:\n"
            "  SizeT: 5")
            ->get_value() == 5);
  // [size_t_argument_base]
  CHECK(TestHelpers::test_factory_creation<BaseClass,
                                           NoGroup<std::unique_ptr<BaseClass>>>(
            "DerivedClassWithoutMetavariables:\n"
            "  SizeT: 6")
            ->get_value() == 6);
  CHECK(
      TestHelpers::test_factory_creation<BaseClass,
                                         OneGroup<std::unique_ptr<BaseClass>>>(
          "DerivedClassWithoutMetavariables:\n"
          "  SizeT: 7")
          ->get_value() == 7);
  CHECK(
      TestHelpers::test_factory_creation<BaseClass,
                                         TwoGroup<std::unique_ptr<BaseClass>>>(
          "DerivedClassWithoutMetavariables:\n"
          "  SizeT: 8")
          ->get_value() == 8);

  // Test derived class that doesn't need metavariables when passing
  // metavariables
  CHECK(TestHelpers::test_factory_creation<
            BaseClass, TestHelpers::TestCreationOpt<std::unique_ptr<BaseClass>>,
            Metavars<4>>("DerivedClassWithoutMetavariables:\n"
                         "  SizeT: 15")
            ->get_value() == 15);
  CHECK(TestHelpers::test_factory_creation<
            BaseClass, NoGroup<std::unique_ptr<BaseClass>>, Metavars<4>>(
            "DerivedClassWithoutMetavariables:\n"
            "  SizeT: 16")
            ->get_value() == 16);
  CHECK(TestHelpers::test_factory_creation<
            BaseClass, OneGroup<std::unique_ptr<BaseClass>>, Metavars<4>>(
            "DerivedClassWithoutMetavariables:\n"
            "  SizeT: 17")
            ->get_value() == 17);
  CHECK(TestHelpers::test_factory_creation<
            BaseClass, TwoGroup<std::unique_ptr<BaseClass>>, Metavars<4>>(
            "DerivedClassWithoutMetavariables:\n"
            "  SizeT: 18")
            ->get_value() == 18);

  // Test creating derived class that requires Metavariables but with no
  // Metavariables passed.
  CHECK(
      TestHelpers::test_factory_creation<
          BaseClass, TestHelpers::TestCreationOpt<std::unique_ptr<BaseClass>>>(
          "DerivedClassWithMetavariables:\n"
          "  SizeT: 4")
          ->get_value() == std::numeric_limits<size_t>::max());
  CHECK(TestHelpers::test_factory_creation<BaseClass,
                                           NoGroup<std::unique_ptr<BaseClass>>>(
            "DerivedClassWithMetavariables:\n"
            "  SizeT: 4")
            ->get_value() == std::numeric_limits<size_t>::max());
  CHECK(
      TestHelpers::test_factory_creation<BaseClass,
                                         OneGroup<std::unique_ptr<BaseClass>>>(
          "DerivedClassWithMetavariables:\n"
          "  SizeT: 4")
          ->get_value() == std::numeric_limits<size_t>::max());
  CHECK(
      TestHelpers::test_factory_creation<BaseClass,
                                         TwoGroup<std::unique_ptr<BaseClass>>>(
          "DerivedClassWithMetavariables:\n"
          "  SizeT: 4")
          ->get_value() == std::numeric_limits<size_t>::max());

  // Test creating derived class passing Metavariables
  CHECK(TestHelpers::test_factory_creation<
            BaseClass, TestHelpers::TestCreationOpt<std::unique_ptr<BaseClass>>,
            Metavars<3>>("DerivedClassWithMetavariables:\n"
                         "  SizeT: 4")
            ->get_value() == 12);
  CHECK(TestHelpers::test_factory_creation<
            BaseClass, NoGroup<std::unique_ptr<BaseClass>>, Metavars<4>>(
            "DerivedClassWithMetavariables:\n"
            "  SizeT: 4")
            ->get_value() == 16);
  CHECK(TestHelpers::test_factory_creation<
            BaseClass, OneGroup<std::unique_ptr<BaseClass>>, Metavars<5>>(
            "DerivedClassWithMetavariables:\n"
            "  SizeT: 4")
            ->get_value() == 20);
  CHECK(TestHelpers::test_factory_creation<
            BaseClass, TwoGroup<std::unique_ptr<BaseClass>>, Metavars<6>>(
            "DerivedClassWithMetavariables:\n"
            "  SizeT: 4")
            ->get_value() == 24);
}

void test_test_enum_creation() {
  // [enum_purple]
  CHECK(TestHelpers::test_creation<Color>("Purple") == Color::Purple);
  // [enum_purple]
  CHECK(TestHelpers::test_creation<Color, TestHelpers::TestCreationOpt<Color>>(
            "Purple") == Color::Purple);
  CHECK(TestHelpers::test_creation<Color, NoGroup<Color>>("Purple") ==
        Color::Purple);
  CHECK(TestHelpers::test_creation<Color, OneGroup<Color>>("Purple") ==
        Color::Purple);
  CHECK(TestHelpers::test_creation<Color, TwoGroup<Color>>("Purple") ==
        Color::Purple);
  CHECK(TestHelpers::test_creation<Color, TestHelpers::TestCreationOpt<Color>,
                                   Metavars<3>>("Purple") == Color::Purple);
  CHECK(TestHelpers::test_creation<Color, NoGroup<Color>, Metavars<3>>(
            "Purple") == Color::Purple);
  CHECK(TestHelpers::test_creation<Color, OneGroup<Color>, Metavars<3>>(
            "Purple") == Color::Purple);
  CHECK(TestHelpers::test_creation<Color, TwoGroup<Color>, Metavars<3>>(
            "Purple") == Color::Purple);
}

SPECTRE_TEST_CASE("Unit.TestCreation", "[Unit]") {
  test_test_creation();
  test_test_factory_creation();
  test_test_enum_creation();
}
}  // namespace
