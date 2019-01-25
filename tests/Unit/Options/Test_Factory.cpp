// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/TMPL.hpp"

namespace {

class OptionTest;
class Test1;
class Test2;
class TestWithArg;
struct TestWithMetavars;

/// [factory_example]
struct OptionType {
  using type = std::unique_ptr<OptionTest>;
  static constexpr OptionString help = {"The type of OptionTest"};
};

class OptionTest {
 public:
  using creatable_classes =
      tmpl::list<Test1, Test2, TestWithArg, TestWithMetavars>;

  OptionTest() = default;
  OptionTest(const OptionTest&) = default;
  OptionTest(OptionTest&&) = default;
  OptionTest& operator=(const OptionTest&) = default;
  OptionTest& operator=(OptionTest&&) = default;
  virtual ~OptionTest() = default;

  virtual std::string name() const = 0;
};

class Test1 : public OptionTest {
 public:
  using options = tmpl::list<>;
  static constexpr OptionString help = {"A derived class"};
  Test1() = default;

  std::string name() const override { return "Test1"; }
};
/// [factory_example]

class Test2 : public OptionTest {
 public:
  using options = tmpl::list<>;
  static constexpr OptionString help = {""};
  Test2() = default;

  std::string name() const override { return "Test2"; }
};

class TestWithArg : public OptionTest {
 public:
  struct Arg {
    using type = std::string;
    static constexpr OptionString help = {"halp"};
  };
  using options = tmpl::list<Arg>;
  static constexpr OptionString help = {""};
  TestWithArg() = default;
  explicit TestWithArg(std::string arg) : arg_(std::move(arg)) {}

  std::string name() const override { return "TestWithArg(" + arg_ + ")"; }

 private:
  std::string arg_;
};

struct Vector {
  using type = std::vector<std::unique_ptr<OptionTest>>;
  static constexpr OptionString help = {"halp"};
};

struct Map {
  using type = std::map<std::string, std::unique_ptr<OptionTest>>;
  static constexpr OptionString help = {"halp"};
};

template <bool Valid>
struct Metavars {
  static constexpr bool valid = Valid;
};

struct TestWithMetavars : OptionTest {
  struct Arg {
    using type = std::string;
    static constexpr OptionString help = {"halp"};
  };
  using options = tmpl::list<Arg>;
  static constexpr OptionString help = {""};

  TestWithMetavars() = default;
  template <typename Metavariables>
  explicit TestWithMetavars(std::string arg, const OptionContext& /*unused*/,
                            Metavariables /*meta*/)
      : arg_(std::move(arg)), valid_(Metavariables::valid) {}

  std::string name() const override {
    return "TestWithArg(" + arg_ + ")" +
           (valid_ ? std::string{"yes"} : std::string{"no"});
  }

 private:
  std::string arg_;
  bool valid_{false};
};

void test_factory() {
  Options<tmpl::list<OptionType>> opts("");
  opts.parse("OptionType: Test2");
  // must pass metavars because TestWithMetavars is a derived class in
  // `creatable_classes`
  CHECK(opts.get<OptionType, Metavars<true>>()->name() == "Test2");
}

void test_factory_with_colon() {
  Options<tmpl::list<OptionType>> opts("");
  opts.parse(
      "OptionType:\n"
      "  Test2:");
  // must pass metavars because TestWithMetavars is a derived class in
  // `creatable_classes`
  CHECK(opts.get<OptionType, Metavars<true>>()->name() == "Test2");
}

void test_factory_with_arg() {
  Options<tmpl::list<OptionType>> opts("");
  opts.parse(
      "OptionType:\n"
      "  TestWithArg:\n"
      "    Arg: stuff");
  // must pass metavars because TestWithMetavars is a derived class in
  // `creatable_classes`
  CHECK(opts.get<OptionType, Metavars<true>>()->name() == "TestWithArg(stuff)");
}

void test_factory_with_metavars() {
  Options<tmpl::list<OptionType>> opts("");
  opts.parse(
      "OptionType:\n"
      "  TestWithMetavars:\n"
      "    Arg: stuff");
  // must pass metavars because TestWithMetavars is a derived class in
  // `creatable_classes`
  CHECK(opts.get<OptionType, Metavars<true>>()->name() ==
        "TestWithArg(stuff)yes");
  // must pass metavars because TestWithMetavars is a derived class in
  // `creatable_classes`
  CHECK(opts.get<OptionType, Metavars<false>>()->name() ==
        "TestWithArg(stuff)no");
}

void test_factory_object_vector() {
  Options<tmpl::list<Vector>> opts("");
  opts.parse("Vector: [Test1, Test2, Test1]");
  // must pass metavars because TestWithMetavars is a derived class in
  // `creatable_classes`
  const auto& arg = opts.get<Vector, Metavars<true>>();
  CHECK(arg.size() == 3);
  CHECK(arg[0]->name() == "Test1");
  CHECK(arg[1]->name() == "Test2");
  CHECK(arg[2]->name() == "Test1");
}

void test_factory_object_map() {
  Options<tmpl::list<Map>> opts("");
  opts.parse(
      "Map:\n"
      "  A: Test1\n"
      "  B: Test2\n"
      "  C: Test1\n");
  // must pass metavars because TestWithMetavars is a derived class in
  // `creatable_classes`
  const auto& arg = opts.get<Map, Metavars<true>>();
  CHECK(arg.size() == 3);
  CHECK(arg.at("A")->name() == "Test1");
  CHECK(arg.at("B")->name() == "Test2");
  CHECK(arg.at("C")->name() == "Test1");
}

void test_factory_format() {
  Options<tmpl::list<OptionType>> opts("");
  INFO(opts.help());
  // The compiler puts "(anonymous namespace)::" before the type, but
  // I don't want to rely on that, so just check that the type is at
  // the end of the line, which should ensure it is not in a template
  // parameter or something.
  CHECK(opts.help().find("OptionTest\n") != std::string::npos);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Options.Factory", "[Unit][Options]") {
  test_factory();
  test_factory_with_arg();
  test_factory_with_colon();
  test_factory_with_metavars();
  test_factory_object_vector();
  test_factory_object_map();
  test_factory_format();
}

// [[OutputRegex, In string:.*At line 1 column 1:.Expected a class to
// create:.Known Ids:.*Test1]]
SPECTRE_TEST_CASE("Unit.Options.Factory.missing", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<OptionType>> opts("");
  opts.parse("OptionType:");
  opts.get<OptionType, Metavars<true>>();
}

// [[OutputRegex, In string:.*At line 2 column 3:.Expected a single class to
// create, got 2]]
SPECTRE_TEST_CASE("Unit.Options.Factory.multiple", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<OptionType>> opts("");
  opts.parse("OptionType:\n"
             "  Test1:\n"
             "  Test2:");
  opts.get<OptionType, Metavars<true>>();
}

// [[OutputRegex, In string:.*At line 1 column 13:.Expected a class or a class
// with options]]
SPECTRE_TEST_CASE("Unit.Options.Factory.vector", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<OptionType>> opts("");
  opts.parse("OptionType: []");
  opts.get<OptionType, Metavars<true>>();
}

// [[OutputRegex, In string:.*At line 1 column 13:.Unknown Id 'Potato']]
SPECTRE_TEST_CASE("Unit.Options.Factory.unknown", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<OptionType>> opts("");
  opts.parse("OptionType: Potato");
  opts.get<OptionType, Metavars<true>>();
}

// [[OutputRegex, In string:.*At line 2 column 1:.You did not specify the
// option 'Arg']]
SPECTRE_TEST_CASE("Unit.Options.Factory.missing_arg", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<OptionType>> opts("");
  opts.parse("OptionType:\n"
             "  TestWithArg:");
  CHECK(opts.get<OptionType, Metavars<true>>()->name() == "TestWithArg(stuff)");
}
