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

/// [factory_example]
struct OptionType {
  using type = std::unique_ptr<OptionTest>;
  static constexpr OptionString help = {"The type of OptionTest"};
};

class OptionTest {
 public:
  using creatable_classes = tmpl::list<Test1, Test2, TestWithArg>;

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
}  // namespace

SPECTRE_TEST_CASE("Unit.Options.Factory", "[Unit][Options]") {
  Options<tmpl::list<OptionType>> opts("");
  opts.parse("OptionType: Test2");
  CHECK(opts.get<OptionType>()->name() == "Test2");
}

SPECTRE_TEST_CASE("Unit.Options.Factory.with_colon", "[Unit][Options]") {
  Options<tmpl::list<OptionType>> opts("");
  opts.parse("OptionType:\n"
             "  Test2:");
  CHECK(opts.get<OptionType>()->name() == "Test2");
}

SPECTRE_TEST_CASE("Unit.Options.Factory.with_arg", "[Unit][Options]") {
  Options<tmpl::list<OptionType>> opts("");
  opts.parse("OptionType:\n"
             "  TestWithArg:\n"
             "    Arg: stuff");
  CHECK(opts.get<OptionType>()->name() == "TestWithArg(stuff)");
}

SPECTRE_TEST_CASE("Unit.Options.Factory.object_vector", "[Unit][Options]") {
  Options<tmpl::list<Vector>> opts("");
  opts.parse("Vector: [Test1, Test2, Test1]");
  const auto& arg = opts.get<Vector>();
  CHECK(arg.size() == 3);
  CHECK(arg[0]->name() == "Test1");
  CHECK(arg[1]->name() == "Test2");
  CHECK(arg[2]->name() == "Test1");
}

SPECTRE_TEST_CASE("Unit.Options.Factory.object_map", "[Unit][Options]") {
  Options<tmpl::list<Map>> opts("");
  opts.parse("Map:\n"
             "  A: Test1\n"
             "  B: Test2\n"
             "  C: Test1\n");
  const auto& arg = opts.get<Map>();
  CHECK(arg.size() == 3);
  CHECK(arg.at("A")->name() == "Test1");
  CHECK(arg.at("B")->name() == "Test2");
  CHECK(arg.at("C")->name() == "Test1");
}

// [[OutputRegex, In string:.*At line 1 column 1:.Expected a class to
// create:.Known Ids:.*Test1]]
SPECTRE_TEST_CASE("Unit.Options.Factory.missing", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<OptionType>> opts("");
  opts.parse("OptionType:");
  opts.get<OptionType>();
}

// [[OutputRegex, In string:.*At line 2 column 3:.Expected a single class to
// create, got 2]]
SPECTRE_TEST_CASE("Unit.Options.Factory.multiple", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<OptionType>> opts("");
  opts.parse("OptionType:\n"
             "  Test1:\n"
             "  Test2:");
  opts.get<OptionType>();
}

// [[OutputRegex, In string:.*At line 1 column 13:.Expected a class or a class
// with options]]
SPECTRE_TEST_CASE("Unit.Options.Factory.vector", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<OptionType>> opts("");
  opts.parse("OptionType: []");
  opts.get<OptionType>();
}

// [[OutputRegex, In string:.*At line 1 column 13:.Unknown Id 'Potato']]
SPECTRE_TEST_CASE("Unit.Options.Factory.unknown", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<OptionType>> opts("");
  opts.parse("OptionType: Potato");
  opts.get<OptionType>();
}

// [[OutputRegex, In string:.*At line 2 column 1:.You did not specify the
// option 'Arg']]
SPECTRE_TEST_CASE("Unit.Options.Factory.missing_arg", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<OptionType>> opts("");
  opts.parse("OptionType:\n"
             "  TestWithArg:");
  CHECK(opts.get<OptionType>()->name() == "TestWithArg(stuff)");
}

SPECTRE_TEST_CASE("Unit.Options.Factory.Format", "[Unit][Options]") {
  Options<tmpl::list<OptionType>> opts("");
  INFO(opts.help());
  // The compiler puts "(anonymous namespace)::" before the type, but
  // I don't want to rely on that, so just check that the type is at
  // the end of the line, which should ensure it is not in a template
  // parameter or something.
  CHECK(opts.help().find("OptionTest\n") != std::string::npos);
}
