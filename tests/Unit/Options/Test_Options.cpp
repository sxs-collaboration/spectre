// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <list>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "Framework/TestHelpers.hpp"
#include "Options/Context.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Options/String.hpp"
#include "Options/Tags.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

namespace {
void test_options_empty() {
  INFO("Empty");
  {
    // Check for no error.
    Options::Parser<tmpl::list<>> opts("");
    opts.parse("");
  }

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<>> opts("");
        opts.parse("Option:");
      }(),
      Catch::Contains(
          "At line 1 column 1:\nOption 'Option' is not a valid option."));

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<>> opts("");
        opts.parse("4");
      }(),
      Catch::Contains("\n'4' does not look like options"));
}

void test_options_syntax_error() {
  INFO("Syntax error");
  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<>> opts("");
        opts.parse(
            "DomainCreator: CreateInterval:\n"
            "  IsPeriodicIn: [false]");
      }(),
      Catch::Contains("At line 1 column 30:\nUnable to correctly parse the "
                      "input file because of a syntax error"));
}

struct Simple {
  using type = int;
  static constexpr Options::String help = {"halp"};
};
struct NamedSimple {
  using type = int;
  static std::string name() { return "SomeName"; }
  static constexpr Options::String help = {
      "halp halp halp halp halp halp halp halp halp halp halp halp\n"
      "halp halp halp halp halp halp halp halp halp halp halp halp"
      "halp halp halp halp halp halp halp halp halp halp halp halp"
      "halp halp halp halp halp halp halp halp halp halp halp halp"
      "halp halp halp halp halp halp halp halp halp halp halp halp"
      "halp halp halp halp halp halp halp halp halp halp halp halp"};
};

void test_options_simple() {
  INFO("Simple");
  {
    Options::Parser<tmpl::list<Simple>> opts("");
    opts.parse("Simple: -4");
    CHECK(opts.get<Simple>() == -4);
  }
  {
    Options::Parser<tmpl::list<NamedSimple>> opts("");
    opts.parse("SomeName: -4");
    CHECK(opts.get<NamedSimple>() == -4);
  }

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<Simple>> opts("");
        opts.parse(
            "Simple: -4\n"
            "Simple: -3");
      }(),
      Catch::Contains("At line 2 column 1:\nOption 'Simple' specified twice."));

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<NamedSimple>> opts("");
        opts.parse(
            "SomeName: -4\n"
            "SomeName: -3");
      }(),
      Catch::Contains(
          "At line 2 column 1:\nOption 'SomeName' specified twice."));

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<Simple>> opts("");
        opts.parse("");
      }(),
      Catch::Contains("In string:\nYou did not specify the option (Simple)"));

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<NamedSimple>> opts("");
        opts.parse("");
      }(),
      Catch::Contains("In string:\nYou did not specify the option (SomeName)"));

  CHECK_THROWS_WITH(
      ([]() {
        Options::Parser<tmpl::list<NamedSimple, Simple>> opts("");
        opts.parse("");
      }()),
      Catch::Contains(
          "In string:\nYou did not specify the options (SomeName,Simple)"));

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<Simple>> opts("");
        opts.parse("Simple:");
        opts.get<Simple>();
      }(),
      Catch::Contains("While parsing option Simple:\nAt line 1 column "
                      "1:\nFailed to convert value to type int:"));

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<NamedSimple>> opts("");
        opts.parse("SomeName:");
        opts.get<NamedSimple>();
      }(),
      Catch::Contains("While parsing option SomeName:\nAt line 1 column "
                      "1:\nFailed to convert value to type int:"));

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<Simple>> opts("");
        opts.parse("Simple: 2.3");
        opts.get<Simple>();
      }(),
      Catch::Contains("While parsing option Simple:\nAt line 1 column "
                      "9:\nFailed to convert value to type int: 2.3"));

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<NamedSimple>> opts("");
        opts.parse("SomeName: 2.3");
        opts.get<NamedSimple>();
      }(),
      Catch::Contains("While parsing option SomeName:\nAt line 1 column "
                      "11:\nFailed to convert value to type int: 2.3"));
}

void test_options_print_long_help() {
  Options::Parser<tmpl::list<NamedSimple>> opts("");
  CHECK(opts.help() ==
        R"(
==== Description of expected options:


Options:
  SomeName:
    type=int
    halp halp halp halp halp halp halp halp halp halp halp halp
    halp halp halp halp halp halp halp halp halp halp halp halphalp halp halp
    halp halp halp halp halp halp halp halp halphalp halp halp halp halp halp
    halp halp halp halp halp halphalp halp halp halp halp halp halp halp halp
    halp halp halphalp halp halp halp halp halp halp halp halp halp halp halp

)");
}

// [options_example_group]
struct Group {
  static constexpr Options::String help = {"Group halp"};
};

struct GroupedTag {
  using type = int;
  static constexpr Options::String help = {"Tag halp"};
  using group = Group;
};
// [options_example_group]

struct OuterGroup {
  static constexpr Options::String help = {"Outer group halp"};
};

struct InnerGroup {
  static constexpr Options::String help = {"Inner group halp"};
  using group = OuterGroup;
};

struct InnerGroupedTag {
  using type = int;
  static constexpr Options::String help = {"Inner tag halp"};
  using group = InnerGroup;
};

struct OuterGroupedTag {
  using type = int;
  static constexpr Options::String help = {"Outer tag halp"};
  using group = OuterGroup;
};

template <typename>
struct TemplatedGroup {
  static constexpr Options::String help = {"halp"};
};

struct TagWithTemplatedGroup {
  using type = int;
  static constexpr Options::String help = {"halp"};
  using group = TemplatedGroup<int>;
};

void test_options_grouped() {
  {
    INFO("Option groups");
    Options::Parser<tmpl::list<GroupedTag, Simple>> opts("Overall help text");
    opts.parse(
        "Group:\n"
        "  GroupedTag: 3\n"
        "Simple: 2");
    CHECK(opts.get<GroupedTag>() == 3);
    CHECK(opts.get<Simple>() == 2);
  }
  {
    INFO("Nested option groups");
    Options::Parser<tmpl::list<InnerGroupedTag, OuterGroupedTag, Simple>> opts(
        "Overall help text");
    opts.parse(
        "OuterGroup:\n"
        "  InnerGroup:\n"
        "    InnerGroupedTag: 3\n"
        "  OuterGroupedTag: 1\n"
        "Simple: 2\n");
    CHECK(opts.get<InnerGroupedTag>() == 3);
    CHECK(opts.get<OuterGroupedTag>() == 1);
    CHECK(opts.get<Simple>() == 2);
  }
  {
    INFO("Templated option groups");
    Options::Parser<tmpl::list<TagWithTemplatedGroup>> opts("");
    opts.parse(
        "TemplatedGroup:\n"
        "  TagWithTemplatedGroup: 3");
    CHECK(opts.get<TagWithTemplatedGroup>() == 3);
  }

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<InnerGroupedTag>> opts("");
        opts.parse("");
        opts.get<InnerGroupedTag>();
      }(),
      Catch::Contains(
          "In string:\nYou did not specify the option (OuterGroup)"));

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<InnerGroupedTag>> opts("");
        opts.parse("OuterGroup:");
        opts.get<InnerGroupedTag>();
      }(),
      Catch::Contains(
          "In group OuterGroup:\nYou did not specify the option (InnerGroup)"));
}

// [options_example_scalar_struct]
struct Bounded {
  using type = int;
  static constexpr Options::String help = {
      "Option with bounds and a suggested value"};
  // These are optional
  static type suggested_value() { return 3; }
  static type lower_bound() { return 2; }
  static type upper_bound() { return 10; }
};
// [options_example_scalar_struct]

struct BadSuggestion {
  using type = int;
  static constexpr Options::String help = {"halp"};
  static type suggested_value() { return 3; }
  static type lower_bound() { return 4; }
};
struct NamedBadSuggestion {
  using type = int;
  static std::string name() { return "SomeName"; }
  static constexpr Options::String help = {"halp"};
  static type suggested_value() { return 3; }
  static type lower_bound() { return 4; }
};

void test_options_suggested() {
  {
    Options::Parser<tmpl::list<Bounded>> opts("Overall help text");
    opts.parse("Bounded: 8");
    CHECK(opts.get<Bounded>() == 8);
  }

  {
    // [options_example_scalar_parse]
    Options::Parser<tmpl::list<Bounded>> opts("Overall help text");
    opts.parse("Bounded: 3");
    CHECK(opts.get<Bounded>() == 3);
    // [options_example_scalar_parse]
  }

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<BadSuggestion>> opts("");
        opts.parse("BadSuggestion: 5");
        opts.get<BadSuggestion>();
      }(),
      Catch::Contains("Checking SUGGESTED value for BadSuggestion:\nValue 3 is "
                      "below the lower bound of 4"));

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<NamedBadSuggestion>> opts("");
        opts.parse("SomeName: 5");
        opts.get<NamedBadSuggestion>();
      }(),
      Catch::Contains("Checking SUGGESTED value for SomeName:\nValue 3 is "
                      "below the lower bound of 4"));
}

// [[OutputRegex, Bounded, line 1:.  Specified: 5.  Suggested: 3]]
SPECTRE_TEST_CASE("Unit.Options.suggestion_warning", "[Unit][Options]") {
  OUTPUT_TEST();
  Options::Parser<tmpl::list<Bounded>> opts("");
  opts.parse("Bounded: 5");
  opts.get<Bounded>();
}

void test_options_bounded() {
  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<Bounded>> opts("");
        opts.parse("Bounded: 1");
        opts.get<Bounded>();
      }(),
      Catch::Contains("While parsing option Bounded:\nAt line 1 column "
                      "10:\nValue 1 is below the lower bound of 2"));

  {
    Options::Parser<tmpl::list<Bounded>> opts("");
    opts.parse("Bounded: 2");
    CHECK(opts.get<Bounded>() == 2);
  }

  {
    Options::Parser<tmpl::list<Bounded>> opts("");
    opts.parse("Bounded: 10");
    CHECK(opts.get<Bounded>() == 10);
  }

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<Bounded>> opts("");
        opts.parse("Bounded: 11");
        opts.get<Bounded>();
      }(),
      Catch::Contains("While parsing option Bounded:\nAt line 1 column "
                      "10:\nValue 11 is above the upper bound of 10"));
}

// [options_example_vector_struct]
struct VectorOption {
  using type = std::vector<int>;
  static constexpr Options::String help = {"A vector with length limits"};
  // These are optional
  static std::string name() {
    return "Vector";  // defaults to "VectorOption"
  }
  static size_t lower_bound_on_size() { return 2; }
  static size_t upper_bound_on_size() { return 5; }
};
// [options_example_vector_struct]

void test_options_bounded_vector() {
  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<VectorOption>> opts("");
        opts.parse("Vector: [2]");
        opts.get<VectorOption>();
      }(),
      Catch::Contains(
          "While parsing option Vector:\nAt line 1 column 9:\nValue must have "
          "at least 2 entries, but 1 were given."));

  {
    Options::Parser<tmpl::list<VectorOption>> opts("");
    opts.parse("Vector: [2,3]");
    CHECK(opts.get<VectorOption>() == (std::vector<int>{2, 3}));
  }

  {
    Options::Parser<tmpl::list<VectorOption>> opts("");
    opts.parse("Vector: [2, 3, 3, 3, 5]");
    CHECK(opts.get<VectorOption>() == (std::vector<int>{2, 3, 3, 3, 5}));
  }

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<VectorOption>> opts("");
        opts.parse("Vector: [2, 3, 3, 3, 5, 6]");
        opts.get<VectorOption>();
      }(),
      Catch::Contains(
          "While parsing option Vector:\nAt line 1 column 9:\nValue must have "
          "at most 5 entries, but 6 were given."));

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<VectorOption>> opts("");
        opts.parse("Vector:");
        opts.get<VectorOption>();
      }(),
      Catch::Contains(
          "While parsing option Vector:\nAt line 1 column 1:\nValue must have "
          "at least 2 entries, but 0 were given."));
}

struct Array {
  using type = std::array<int, 3>;
  static constexpr Options::String help = {"halp"};
};

struct ZeroArray {
  using type = std::array<int, 0>;
  static constexpr Options::String help = {"halp"};
};

void test_options_array() {
  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<Array>> opts("");
        opts.parse("Array: [1, 2]");
        opts.get<Array>();
      }(),
      Catch::Contains("While parsing option Array:\nAt line 1 column "
                      "8:\nFailed to convert value to type [int x3]: [1, 2]"));

  {
    Options::Parser<tmpl::list<Array>> opts("");
    opts.parse("Array: [1,2,3]");
    CHECK(opts.get<Array>() == (std::array<int, 3>{{1, 2, 3}}));
  }

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<Array>> opts("");
        opts.parse("Array: [1, 2, 3, 4]");
        opts.get<Array>();
      }(),
      Catch::Contains(
          "While parsing option Array:\nAt line 1 column 8:\nFailed to convert "
          "value to type [int x3]: [1, 2, 3, 4]"));

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<Array>> opts("");
        opts.parse(
            "Array:\n"
            "  - 1\n"
            "  - 2\n"
            "  - 3\n"
            "  - 4");
        opts.get<Array>();
      }(),
      Catch::Contains(
          "While parsing option Array:\nAt line 2 column 3:\nFailed to convert "
          "value to type [int x3]:\n  - 1\n  - 2\n  - 3\n  - 4"));

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<Array>> opts("");
        opts.parse("Array:");
        opts.get<Array>();
      }(),
      Catch::Contains("While parsing option Array:\nAt line 1 column "
                      "1:\nFailed to convert value to type [int x3]:"));

  {
    Options::Parser<tmpl::list<ZeroArray>> opts("");
    opts.parse("ZeroArray:");
    opts.get<ZeroArray>();
  }
}

struct Map {
  using type = std::map<std::string, int>;
  static constexpr Options::String help = {"halp"};
};

void test_options_map() {
  {
    Options::Parser<tmpl::list<Map>> opts("");
    opts.parse(
        "Map:\n"
        "  A: 3\n"
        "  Z: 2");
    std::map<std::string, int> expected;
    expected.emplace("A", 3);
    expected.emplace("Z", 2);
    CHECK(opts.get<Map>() == expected);
  }

  {
    Options::Parser<tmpl::list<Map>> opts("");
    opts.parse("Map:");
    CHECK(opts.get<Map>().empty());
  }

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<Map>> opts("");
        opts.parse("Map: string");
        opts.get<Map>();
      }(),
      Catch::Contains("While parsing option Map:\nAt line 1 column 6:\nFailed "
                      "to convert value to type {string: int}: string"));

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<Map>> opts("");
        opts.parse(
            "Map:\n"
            "  A: string");
        opts.get<Map>();
      }(),
      Catch::Contains("While parsing option Map:\nAt line 2 column 6:\nFailed "
                      "to convert value to type {string: int}: A: string"));
}

struct UnorderedMap {
  using type = std::unordered_map<std::string, int>;
  static constexpr Options::String help = {"halp"};
};

void test_options_unordered_map() {
  Options::Parser<tmpl::list<UnorderedMap>> opts("");
  opts.parse(
      "UnorderedMap:\n"
      "  A: 3\n"
      "  Z: 2");
  std::unordered_map<std::string, int> expected;
  expected.emplace("A", 3);
  expected.emplace("Z", 2);
  CHECK(opts.get<UnorderedMap>() == expected);
}

template <typename... T>
struct VariantTag {
  using type = std::variant<T...>;
  static constexpr Options::String help = {"halp"};
};

struct VariantOption1 {
  static constexpr Options::String help = {"VariantOption1 halp"};
  struct Opt1 {
    using type = int;
    static constexpr Options::String help = {"halp"};
  };
  using options = tmpl::list<Opt1>;
  VariantOption1() = default;
  VariantOption1(int /*unused*/) {}
};

struct VariantOption2 {
  static constexpr Options::String help = {"VariantOption2 halp"};
  struct Opt2 {
    using type = int;
    static constexpr Options::String help = {"halp"};
  };
  template <typename Metavariables>
  using options =
      tmpl::list<tmpl::conditional_t<Metavariables::valid, Opt2, void>>;
  VariantOption2() = default;
  VariantOption2(tmpl::list<Opt2> /*meta*/, int /*unused*/) {}
};

struct VariantOption2Metavars {
  static constexpr bool valid = true;
};

struct VariantOptionWithGroup {
  static constexpr Options::String help = {"VariantOptionWithGroup halp"};
  struct Group {
    static constexpr Options::String help = {"halp"};
  };
  struct Opt3 {
    using type = int;
    static constexpr Options::String help = {"halp"};
    using group = Group;
  };
  using options = tmpl::list<Opt3>;
  VariantOptionWithGroup() = default;
  VariantOptionWithGroup(int /*unused*/) {}
};

void test_options_variant() {
  {
    const auto check = [](const std::string& input, const auto& expected) {
      using Tag = tmpl::wrap<std::decay_t<decltype(expected)>, VariantTag>;
      Options::Parser<tmpl::list<Tag>> opts("");
      opts.parse("VariantTag: " + input);
      CHECK(opts.template get<Tag>() == expected);
    };
    check("3", std::variant<int>(std::in_place_type_t<int>{}, 3));
    check("Hello", std::variant<std::string>(
                       std::in_place_type_t<std::string>{}, "Hello"));
    check("3", std::variant<int, std::string>(std::in_place_type_t<int>{}, 3));
    check("Hello", std::variant<int, std::string>(
                       std::in_place_type_t<std::string>{}, "Hello"));
    check("3", std::variant<std::string, int>(
                   std::in_place_type_t<std::string>{}, "3"));
    check("Hello", std::variant<std::string, int>(
                       std::in_place_type_t<std::string>{}, "Hello"));

    {
      const std::string help_text =
          Options::Parser<tmpl::list<VariantTag<int, std::string>>>("").help();
      CAPTURE(help_text);
      CHECK(help_text.find("type=int or string\n") != std::string::npos);
    }
  }

  {
    using tag = VariantTag<int, std::string>;
    CHECK_THROWS_WITH(
        []() {
          Options::Parser<tmpl::list<tag>> opts("");
          opts.parse("VariantTag: []");
          opts.get<tag>();
        }(),
        Catch::Contains("While creating a variant:\nAt line 1 column "
                        "13:\nFailed to convert value to type int or string: "
                        "[]") and
            Catch::Contains("At line 1 column 13:\nFailed to convert value to "
                            "type int: []") and
            Catch::Contains("At line 1 column 13:\nFailed to convert value to "
                            "type string: []"));
  }

  {
    using tag = VariantTag<VariantOption1, VariantOption2>;
    {
      Options::Parser<tmpl::list<tag>> parser("halp");
      parser.parse(
          "VariantTag:\n"
          "  Opt1: 3");
      CHECK(std::holds_alternative<VariantOption1>(
          parser.get<tag, VariantOption2Metavars>()));
    }
    {
      Options::Parser<tmpl::list<tag>> parser("halp");
      parser.parse(
          "VariantTag:\n"
          "  Opt2: 3");
      CHECK(std::holds_alternative<VariantOption2>(
          parser.get<tag, VariantOption2Metavars>()));
    }
    CHECK_THROWS_WITH(([]() {
                        Options::Parser<tmpl::list<tag>> parser("halp");
                        parser.parse(
                            "VariantTag:\n"
                            "  OptZ: 3");
                        parser.get<tag, VariantOption2Metavars>();
                      }()),
                      Catch::Contains("VariantOption1 halp") and
                          Catch::Contains("VariantOption2 halp") and
                          Catch::Contains("EITHER") and
                          not Catch::Contains("Possible errors"));
  }
  {
    using tag = VariantTag<VariantOption1, int>;
    {
      Options::Parser<tmpl::list<tag>> parser("halp");
      parser.parse(
          "VariantTag:\n"
          "  Opt1: 3");
      CHECK(std::holds_alternative<VariantOption1>(parser.get<tag>()));
    }
    {
      Options::Parser<tmpl::list<tag>> parser("halp");
      parser.parse("VariantTag: 3");
      CHECK(std::holds_alternative<int>(parser.get<tag>()));
    }
    CHECK_THROWS_WITH(
        ([]() {
          Options::Parser<tmpl::list<tag>> parser("halp");
          parser.parse(
              "VariantTag:\n"
              "  OptZ: 3");
          parser.get<tag>();
        }()),
        Catch::Contains(
            "Failed to convert value to type VariantOption1 or int:") and
            Catch::Contains("VariantOption1 halp") and
            Catch::Contains("Failed to convert value to type int:") and
            not Catch::Contains("EITHER") and
            Catch::Contains("Possible errors"));
  }
  {
    using tag = VariantTag<VariantOption1, int, VariantOption2>;
    {
      Options::Parser<tmpl::list<tag>> parser("halp");
      parser.parse(
          "VariantTag:\n"
          "  Opt1: 3");
      CHECK(std::holds_alternative<VariantOption1>(
          parser.get<tag, VariantOption2Metavars>()));
    }
    {
      Options::Parser<tmpl::list<tag>> parser("halp");
      parser.parse("VariantTag: 3");
      CHECK(std::holds_alternative<int>(
          parser.get<tag, VariantOption2Metavars>()));
    }
    {
      Options::Parser<tmpl::list<tag>> parser("halp");
      parser.parse(
          "VariantTag:\n"
          "  Opt2: 3");
      CHECK(std::holds_alternative<VariantOption2>(
          parser.get<tag, VariantOption2Metavars>()));
    }
    CHECK_THROWS_WITH(
        ([]() {
          Options::Parser<tmpl::list<tag>> parser("halp");
          parser.parse(
              "VariantTag:\n"
              "  OptZ: 3");
          parser.get<tag, VariantOption2Metavars>();
        }()),
        Catch::Contains("Failed to convert value to type VariantOption1 or int "
                        "or VariantOption2:") and
            Catch::Contains("VariantOption1 halp") and
            Catch::Contains("VariantOption2 halp") and
            Catch::Contains("Failed to convert value to type int:") and
            Catch::Contains("EITHER") and Catch::Contains("Possible errors"));
  }
  {
    using tag =
        VariantTag<VariantOption1, VariantOption2, VariantOptionWithGroup>;
    {
      Options::Parser<tmpl::list<tag>> parser("halp");
      parser.parse(
          "VariantTag:\n"
          "  Opt1: 3");
      CHECK(std::holds_alternative<VariantOption1>(
          parser.get<tag, VariantOption2Metavars>()));
    }
    {
      Options::Parser<tmpl::list<tag>> parser("halp");
      parser.parse(
          "VariantTag:\n"
          "  Opt2: 3");
      CHECK(std::holds_alternative<VariantOption2>(
          parser.get<tag, VariantOption2Metavars>()));
    }
    {
      Options::Parser<tmpl::list<tag>> parser("halp");
      parser.parse(
          "VariantTag:\n"
          "  Group:\n"
          "    Opt3: 3");
      CHECK(std::holds_alternative<VariantOptionWithGroup>(
          parser.get<tag, VariantOption2Metavars>()));
    }
    CHECK_THROWS_WITH(
        ([]() {
          Options::Parser<tmpl::list<tag>> parser("halp");
          parser.parse(
              "VariantTag:\n"
              "  OptZ: 3");
          parser.get<tag, VariantOption2Metavars>();
        }()),
        Catch::Contains("Failed to convert value to type VariantOption1 or "
                        "VariantOption2 or VariantOptionWithGroup:") and
            Catch::Contains("VariantOption1 halp") and
            Catch::Contains("VariantOption2 halp") and
            Catch::Contains("VariantOptionWithGroup halp") and
            Catch::Contains("EITHER") and Catch::Contains("Possible errors"));
  }
}

template <typename T>
struct Wrapped {
  T data;
};

#define FORWARD_OP(op)                                         \
  template <typename T>                                        \
  bool operator op(const Wrapped<T>& a, const Wrapped<T>& b) { \
    return a.data op b.data;                                   \
  }
FORWARD_OP(==)
FORWARD_OP(!=)
FORWARD_OP(<)
FORWARD_OP(>)
FORWARD_OP(<=)
FORWARD_OP(>=)
}  // namespace

namespace std {
template <typename T>
struct hash<Wrapped<T>> {
  size_t operator()(const Wrapped<T>& x) const { return hash<T>{}(x.data); }
};
}  // namespace std

template <typename T>
struct Options::create_from_yaml<Wrapped<T>> {
  template <typename Metavariables>
  static Wrapped<T> create(const Options::Option& options) {
    return Wrapped<T>{options.parse_as<T>()};
  }
};

namespace {
struct WrapMap {
  using type = std::map<Wrapped<int>, Wrapped<std::string>>;
  static constexpr Options::String help = {"halp"};
};
struct WrapVector {
  using type = std::vector<Wrapped<int>>;
  static constexpr Options::String help = {"halp"};
};
struct WrapList {
  using type = std::list<Wrapped<int>>;
  static constexpr Options::String help = {"halp"};
};
struct WrapArray {
  using type = std::array<Wrapped<int>, 2>;
  static constexpr Options::String help = {"halp"};
};
struct WrapPair {
  using type = std::pair<Wrapped<int>, Wrapped<std::string>>;
  static constexpr Options::String help = {"halp"};
};
struct WrapUnorderedMap {
  using type = std::unordered_map<Wrapped<int>, Wrapped<std::string>>;
  static constexpr Options::String help = {"halp"};
};

void test_options_complex_containers() {
  Options::Parser<tmpl::list<WrapMap, WrapVector, WrapList, WrapArray, WrapPair,
                             WrapUnorderedMap>>
      opts("");
  opts.parse(
      "WrapMap: {1: A, 2: B}\n"
      "WrapVector: [1, 2, 3]\n"
      "WrapList: [1, 2, 3]\n"
      "WrapArray: [1, 2]\n"
      "WrapPair: [1, X]\n"
      "WrapUnorderedMap: {1: A, 2: B}\n");
  CHECK(opts.get<WrapMap>() == (std::map<Wrapped<int>, Wrapped<std::string>>{
                                   {{1}, {"A"}}, {{2}, {"B"}}}));
  CHECK(opts.get<WrapVector>() == (std::vector<Wrapped<int>>{{1}, {2}, {3}}));
  CHECK(opts.get<WrapList>() == (std::list<Wrapped<int>>{{1}, {2}, {3}}));
  CHECK(opts.get<WrapArray>() == (std::array<Wrapped<int>, 2>{{{1}, {2}}}));
  CHECK(opts.get<WrapPair>() ==
        (std::pair<Wrapped<int>, Wrapped<std::string>>{{1}, {"X"}}));
  CHECK(opts.get<WrapUnorderedMap>() ==
        (std::unordered_map<Wrapped<int>, Wrapped<std::string>>{{{1}, {"A"}},
                                                                {{2}, {"B"}}}));
}

#ifdef SPECTRE_DEBUG
struct Duplicate {
  using type = int;
  static constexpr Options::String help = {"halp"};
};
struct NamedDuplicate {
  using type = int;
  static std::string name() { return "Duplicate"; }
  static constexpr Options::String help = {"halp"};
};

struct
    ToooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooLong {
  using type = int;
  static constexpr Options::String help = {"halp"};
};
using short_alias_for_too_long =
    ToooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooLong;
struct NamedTooLong {
  using type = int;
  static std::string name() {
    return "Toooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo"
           "oLong";
  }
  static constexpr Options::String help = {"halp"};
};
struct NoHelp {
  using type = int;
  static constexpr Options::String help = {""};
};
struct NamedNoHelp {
  using type = int;
  static std::string name() { return "NoHelp"; }
  static constexpr Options::String help = {""};
};
#endif

void test_options_invalid_calls() {
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      ([]() {
        Options::Parser<tmpl::list<Duplicate, NamedDuplicate>> opts("");
        opts.parse("");
      }()),
      Catch::Contains("Duplicate option name: Duplicate"));

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<short_alias_for_too_long>> opts("");
      }(),
      Catch::Contains("The option name "
                      "Tooooooooooooooooooooooooooooooooooooooooooooooooooooo"
                      "ooooooooooooooLong is too long for nice formatting"));
  CHECK_THROWS_WITH(
      []() { Options::Parser<tmpl::list<NamedTooLong>> opts(""); }(),
      Catch::Contains("The option name "
                      "Tooooooooooooooooooooooooooooooooooooooooooooooooooooo"
                      "ooooooooooooooLong is too long for nice formatting"));

  CHECK_THROWS_WITH(
      Options::Parser<tmpl::list<NoHelp>>(""),
      Catch::Contains(
          "You must supply a help string of non-zero length for NoHelp"));

  CHECK_THROWS_WITH(
      Options::Parser<tmpl::list<NamedNoHelp>>(""),
      Catch::Contains(
          "You must supply a help string of non-zero length for NoHelp"));
#endif
}

struct Apply1 {
  using type = int;
  static constexpr Options::String help = {"halp"};
};
struct Apply2 {
  using type = std::string;
  static constexpr Options::String help = {"halp"};
};
struct Apply3 {
  using type = std::vector<int>;
  static constexpr Options::String help = {"halp"};
};

void test_options_apply() {
  Options::Parser<tmpl::list<Apply1, Apply2, Apply3>> opts("");
  opts.parse(
      "Apply1: 2\n"
      "Apply2: str\n"
      "Apply3: [1, 2, 3]");
  // We do the checks outside the lambda to make sure it actually gets called.
  std::vector<int> arg1;
  int arg2 = 0;
  opts.apply<tmpl::list<Apply3, Apply1>>([&](const auto& a, auto b) {
    arg1 = a;
    arg2 = b;
  });
  CHECK(arg1 == (std::vector<int>{1, 2, 3}));
  CHECK(arg2 == 2);
}

void test_options_option_context_default_stream() {
  CHECK(get_output(Options::Context{}).empty());
}

// Use formatted inner types to make sure nested formatting works
template <typename T>
using In = std::array<T, 0>;
struct FormatMap {
  using type = std::map<In<int>, In<double>>;
  static constexpr Options::String help = {"halp"};
  static constexpr const char* const expected = "{[int x0]: [double x0]}";
};
struct FormatVector {
  using type = std::vector<In<int>>;
  static constexpr Options::String help = {"halp"};
  static constexpr const char* const expected = "[[int x0], ...]";
};
struct FormatList {
  using type = std::list<In<int>>;
  static constexpr Options::String help = {"halp"};
  static constexpr const char* const expected = "[[int x0], ...]";
};
struct FormatArray {
  using type = std::array<In<int>, 3>;
  static constexpr Options::String help = {"halp"};
  static constexpr const char* const expected = "[[int x0] x3]";
};
struct FormatPair {
  using type = std::pair<In<int>, In<double>>;
  static constexpr Options::String help = {"halp"};
  static constexpr const char* const expected = "[[int x0], [double x0]]";
};
struct ArrayHash {
  size_t operator()(const std::array<int, 0>& /*unused*/) const { return 0; }
};
struct FormatUnorderedMap {
  using type = std::unordered_map<In<int>, In<double>, ArrayHash>;
  static constexpr Options::String help = {"halp"};
  static constexpr const char* const expected = "{[int x0]: [double x0]}";
};

struct ScalarWithLimits {
  using type = int;
  static constexpr Options::String help = "ScalarHelp";
  static type suggested_value() { return 7; }
  static type lower_bound() { return 2; }
  static type upper_bound() { return 8; }
};

struct VectorWithLimits {
  using type = std::vector<int>;
  static constexpr Options::String help = "VectorHelp";
  static size_t lower_bound_on_size() { return 5; }
  static size_t upper_bound_on_size() { return 9; }
};

struct SuggestedBool {
  using type = bool;
  static type suggested_value() { return false; }
  constexpr static Options::String help = "halp";
};

void test_options_format() {
  {
    const auto check = [](auto opt) {
      using Opt = decltype(opt);
      Options::Parser<tmpl::list<Opt>> opts("");
      INFO("Help string:\n"
           << opts.help() << "\n\nExpected to find:\n"
           << "  type="s + Opt::expected + "\n");
      // Add whitespace to check that we've got the entire type
      CHECK(opts.help().find("type="s + Opt::expected + "\n"s) !=
            std::string::npos);
    };
    check(FormatMap{});
    check(FormatVector{});
    check(FormatList{});
    check(FormatArray{});
    check(FormatPair{});
    check(FormatUnorderedMap{});
  }

  CHECK(Options::Parser<tmpl::list<ScalarWithLimits>>("").help() == R"(
==== Description of expected options:


Options:
  ScalarWithLimits:
    type=int
    suggested=7
    min=2
    max=8
    ScalarHelp

)");
  CHECK(Options::Parser<tmpl::list<VectorWithLimits>>("").help() == R"(
==== Description of expected options:


Options:
  VectorWithLimits:
    type=[int, ...]
    min size=5
    max size=9
    VectorHelp

)");

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<FormatUnorderedMap>> opts("");
        opts.parse("FormatUnorderedMap: X");
        opts.get<FormatUnorderedMap>();
      }(),
      Catch::Contains(
          "Failed to convert value to type {[int x0]: [double x0]}:"));

  {
    const auto help = Options::Parser<tmpl::list<SuggestedBool>>("").help();
    CAPTURE(help);
    CHECK(help.find("suggested=false") != std::string::npos);
  }
}

struct ExplicitObject {
  explicit ExplicitObject() = default;
};

struct ExplicitObjectTag {
  using type = ExplicitObject;
  static constexpr Options::String help = {"halp"};
};
}  // namespace

template <>
struct Options::create_from_yaml<ExplicitObject> {
  template <typename Metavariables>
  static ExplicitObject create(const Options::Option& /*options*/) {
    return ExplicitObject{};
  }
};

namespace {
void test_options_explicit_constructor() {
  Options::Parser<tmpl::list<ExplicitObjectTag>> opts("");
  opts.parse("ExplicitObjectTag:");
  opts.get<ExplicitObjectTag>();
}

void test_options_input_source() {
  Options::Parser<tmpl::list<Simple>> parser("");
  const std::string source = "Simple: 3";
  const std::string overlay = "Simple: 4";
  parser.parse(source);
  parser.overlay<tmpl::list<Simple>>(overlay);
  CHECK(parser.get<Options::Tags::InputSource>() ==
        std::vector{source, overlay});
}

void check_for_lines(const std::string& text,
                     const std::vector<std::string>& lines) {
  CAPTURE(text);
  const std::string search_text = "\n" + text + "\n";
  size_t pos = 0;
  for (const auto& line : lines) {
    CAPTURE(line);
    pos = search_text.find("\n" + line + "\n");
    if (pos == search_text.npos) {
      CHECK(false);
      // After failing to find a line none of the remaining lines will
      // be found after it (because we are at the end of the string),
      // so avoid printing an error for each of them.
      break;
    }
  }
}

struct Alternatives {
  struct A {
    using type = double;
    static constexpr Options::String help = "halp";
  };

  struct B {
    using type = int;
    static constexpr Options::String help = "halp";
  };

  struct C {
    using type = int;
    static constexpr Options::String help = "halp";
  };

  struct D {
    using type = std::vector<int>;
    static constexpr Options::String help = "halp";
  };

  struct E {
    using type = bool;
    static constexpr Options::String help = "halp";
  };

  struct F {
    using type = std::string;
    static constexpr Options::String help = "halp";
  };

  static constexpr Options::String help = "halp";
  Alternatives() = default;

  // [alternatives]
  using options = tmpl::list<
      A, Options::Alternatives<tmpl::list<B>, tmpl::list<C>, tmpl::list<D, E>>,
      F>;
  Alternatives(tmpl::list<A, B, F> /*meta*/, double a, int b, std::string f)
      : a_(a), b_(b), f_(std::move(f)) {}
  Alternatives(tmpl::list<A, C, F> /*meta*/, double a, int c, std::string f)
      : a_(a), c_(c), f_(std::move(f)) {}
  Alternatives(double a, std::vector<int> d, bool e, std::string f)
      : a_(a), d_(std::move(d)), e_(e), f_(std::move(f)) {}
  // [alternatives]

  double a_{-1.0};
  int b_{-1};
  int c_{-1};
  std::vector<int> d_{};
  bool e_{false};
  std::string f_{};
};

struct AlternativesTag {
  using type = Alternatives;
  static constexpr Options::String help = "halp";
};

struct NestedAlternatives {
  struct A {
    using type = int;
    static constexpr Options::String help = "halp";
  };

  struct B {
    using type = double;
    static constexpr Options::String help = "halp";
  };

  struct C {
    using type = std::string;
    static constexpr Options::String help = "halp";
  };

  struct D {
    using type = std::vector<int>;
    static constexpr Options::String help = "halp";
  };

  struct E {
    using type = bool;
    static constexpr Options::String help = "halp";
  };

  static constexpr Options::String help = "halp";
  using options = tmpl::list<Options::Alternatives<
      tmpl::list<A, Options::Alternatives<tmpl::list<B>, tmpl::list<C>>>,
      tmpl::list<D, E>>>;

  NestedAlternatives() = default;
  NestedAlternatives(int a, double b) : a_(a), b_(b) {}
  NestedAlternatives(int a, std::string c) : a_(a), c_(std::move(c)) {}
  NestedAlternatives(std::vector<int> d, bool e) : d_(std::move(d)), e_(e) {}

  int a_{-1};
  double b_{-1.0};
  std::string c_{};
  std::vector<int> d_{};
  bool e_{false};
};

struct NestedAlternativesTag {
  using type = NestedAlternatives;
  static constexpr Options::String help = "halp";
};

void test_options_alternatives() {
  {
    Options::Parser<tmpl::list<AlternativesTag>> parser("");
    parser.parse(
        "AlternativesTag:\n"
        "  A: 7.8\n"
        "  B: 4\n"
        "  F: first\n");
    const auto result = parser.get<AlternativesTag>();
    CHECK(result.a_ == 7.8);
    CHECK(result.b_ == 4);
    CHECK(result.c_ == -1);
    CHECK(result.d_.empty());
    CHECK(result.e_ == false);
    CHECK(result.f_ == "first");
  }
  {
    Options::Parser<tmpl::list<AlternativesTag>> parser("");
    parser.parse(
        "AlternativesTag:\n"
        "  C: 5\n"
        "  A: 7.8\n"
        "  F: second\n");
    const auto result = parser.get<AlternativesTag>();
    CHECK(result.a_ == 7.8);
    CHECK(result.b_ == -1);
    CHECK(result.c_ == 5);
    CHECK(result.d_.empty());
    CHECK(result.e_ == false);
    CHECK(result.f_ == "second");
  }
  {
    Options::Parser<tmpl::list<AlternativesTag>> parser("");
    parser.parse(
        "AlternativesTag:\n"
        "  A: 7.8\n"
        "  D: [2, 3, 5, 7, 11]\n"
        "  E: true\n"
        "  F: third\n");
    const auto result = parser.get<AlternativesTag>();
    CHECK(result.a_ == 7.8);
    CHECK(result.b_ == -1);
    CHECK(result.c_ == -1);
    CHECK(result.d_ == std::vector<int>{2, 3, 5, 7, 11});
    CHECK(result.e_ == true);
    CHECK(result.f_ == "third");
  }

  {
    Options::Parser<tmpl::list<NestedAlternativesTag>> parser("");
    parser.parse(
        "NestedAlternativesTag:\n"
        "  A: 3\n"
        "  B: 7.8\n");
    const auto result = parser.get<NestedAlternativesTag>();
    CHECK(result.a_ == 3);
    CHECK(result.b_ == 7.8);
    CHECK(result.c_.empty());
    CHECK(result.d_.empty());
    CHECK(result.e_ == false);
  }
  {
    Options::Parser<tmpl::list<NestedAlternativesTag>> parser("");
    parser.parse(
        "NestedAlternativesTag:\n"
        "  C: hi\n"
        "  A: 3\n");
    const auto result = parser.get<NestedAlternativesTag>();
    CHECK(result.a_ == 3);
    CHECK(result.b_ == -1.0);
    CHECK(result.c_ == "hi");
    CHECK(result.d_.empty());
    CHECK(result.e_ == false);
  }
  {
    Options::Parser<tmpl::list<NestedAlternativesTag>> parser("");
    parser.parse(
        "NestedAlternativesTag:\n"
        "  D: [2, 3, 5, 7, 11]\n"
        "  E: true\n");
    const auto result = parser.get<NestedAlternativesTag>();
    CHECK(result.a_ == -1);
    CHECK(result.b_ == -1.0);
    CHECK(result.c_.empty());
    CHECK(result.d_ == std::vector<int>{2, 3, 5, 7, 11});
    CHECK(result.e_ == true);
  }

  // clang-format off
  check_for_lines(Options::Parser<Alternatives::options>("").help(),
                  {"Options:",
                   "  A:",
                   "  EITHER",
                   "    B:",
                   "  OR",
                   "    C:",
                   "  OR",
                   "    D:",
                   "    E:",
                   "  F:"});
  check_for_lines(Options::Parser<NestedAlternatives::options>("").help(),
                  {"Options:",
                   "  EITHER",
                   "    A:",
                   "    EITHER",
                   "      B:",
                   "    OR",
                   "      C:",
                   "  OR",
                   "    D:",
                   "    E:"});
  // clang-format on
}

void test_options_overlay() {
  {
    Options::Parser<tmpl::list<Simple, InnerGroupedTag, OuterGroupedTag>>
        parser("");
    parser.parse(
        "Simple: 1\n"
        "OuterGroup:\n"
        "  OuterGroupedTag: 2\n"
        "  InnerGroup:\n"
        "    InnerGroupedTag: 3\n");
    CHECK(parser.get<Simple>() == 1);
    CHECK(parser.get<OuterGroupedTag>() == 2);
    CHECK(parser.get<InnerGroupedTag>() == 3);
    parser.overlay<tmpl::list<Simple>>("Simple: 4");
    CHECK(parser.get<Simple>() == 4);
    CHECK(parser.get<OuterGroupedTag>() == 2);
    CHECK(parser.get<InnerGroupedTag>() == 3);
    parser.overlay<tmpl::list<Simple>>("");
    CHECK(parser.get<Simple>() == 4);
    CHECK(parser.get<OuterGroupedTag>() == 2);
    CHECK(parser.get<InnerGroupedTag>() == 3);
    parser.overlay<tmpl::list<InnerGroupedTag>>(
        "OuterGroup:\n"
        "  InnerGroup:\n"
        "    InnerGroupedTag: 5\n");
    CHECK(parser.get<Simple>() == 4);
    CHECK(parser.get<OuterGroupedTag>() == 2);
    CHECK(parser.get<InnerGroupedTag>() == 5);
    parser.overlay<tmpl::list<Simple, OuterGroupedTag>>("Simple: 6\n");
    CHECK(parser.get<Simple>() == 6);
    CHECK(parser.get<OuterGroupedTag>() == 2);
    CHECK(parser.get<InnerGroupedTag>() == 5);
    parser.overlay<tmpl::list<Simple, OuterGroupedTag>>(
        "OuterGroup:\n"
        "  OuterGroupedTag: 7\n");
    CHECK(parser.get<Simple>() == 6);
    CHECK(parser.get<OuterGroupedTag>() == 7);
    CHECK(parser.get<InnerGroupedTag>() == 5);
    parser.overlay<tmpl::list<Simple, OuterGroupedTag>>(
        "Simple: 8\n"
        "OuterGroup:\n"
        "  OuterGroupedTag: 9\n");
    CHECK(parser.get<Simple>() == 8);
    CHECK(parser.get<OuterGroupedTag>() == 9);
    CHECK(parser.get<InnerGroupedTag>() == 5);
  }

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<Simple>> parser("");
        parser.parse("Simple: 1");
        parser.overlay<tmpl::list<Simple>>("NotSimple: 2");
      }(),
      Catch::Contains(
          "At line 1 column 1:\nOption 'NotSimple' is not a valid option."));

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<Simple>> parser("");
        parser.parse("Simple: 1");
        parser.overlay<tmpl::list<>>("Simple: 2");
      }(),
      Catch::Contains(
          "At line 1 column 1:\nOption 'Simple' is not overlayable."));

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<Simple>> parser("");
        parser.parse("Simple: 1");
        parser.overlay<tmpl::list<Simple>>(
            "Simple: 2\n"
            "Simple: 2");
      }(),
      Catch::Contains("At line 2 column 1:\nOption 'Simple' specified twice."));

  CHECK_THROWS_WITH(
      []() {
        Options::Parser<tmpl::list<OuterGroupedTag>> parser("");
        parser.parse(
            "OuterGroup:\n"
            "  OuterGroupedTag: 1");
        parser.overlay<tmpl::list<OuterGroupedTag>>(
            "OuterGroup:\n"
            "  OuterGroupedTag: 2\n"
            "  OuterGroupedTag: 2");
      }(),
      Catch::Contains("In group OuterGroup:\nAt line 3 column 3:\nOption "
                      "'OuterGroupedTag' specified twice."));
}

void test_options_serialization() {
  {
    // Test serialization of an unparsed parser.
    Options::Parser<tmpl::list<Simple>> parser("passed help");
    auto parser2 = serialize_and_deserialize(parser);
    CHECK(parser2.get<Options::Tags::InputSource>().empty());
    CHECK(parser2.help() == parser.help());
    const std::string source = "Simple: 4";
    parser2.parse(source);
    const auto parser3 = serialize_and_deserialize(parser2);
    CHECK(parser3.get<Options::Tags::InputSource>() == std::vector{source});
    CHECK(parser3.get<Simple>() == 4);
  }

  const auto check_repeated = [](const auto& parser1, const auto& f) {
    f(parser1);
    const auto parser2 = serialize_and_deserialize(parser1);
    f(parser2);
    const auto parser3 = serialize_and_deserialize(parser2);
    f(parser3);
  };

  {
    // Groups
    Options::Parser<tmpl::list<InnerGroupedTag, OuterGroupedTag, Simple>>
        parser("Overall help text");
    const std::string source =
        "OuterGroup:\n"
        "  InnerGroup:\n"
        "    InnerGroupedTag: 3\n"
        "  OuterGroupedTag: 1\n"
        "Simple: 2\n";
    parser.parse(source);
    check_repeated(parser, [&source](const decltype(parser)& local_parser) {
      CHECK(local_parser.get<Options::Tags::InputSource>() ==
            std::vector{source});
      CHECK(local_parser.get<InnerGroupedTag>() == 3);
      CHECK(local_parser.get<OuterGroupedTag>() == 1);
      CHECK(local_parser.get<Simple>() == 2);
    });
  }
  {
    // Overlays
    Options::Parser<tmpl::list<Simple>> parser("");
    const std::string source = "Simple: 3";
    const std::string overlay = "Simple: 4";
    parser.parse(source);
    parser.overlay<tmpl::list<Simple>>(overlay);
    check_repeated(parser,
                   [&source, &overlay](const decltype(parser)& local_parser) {
                     CHECK(local_parser.get<Options::Tags::InputSource>() ==
                           std::vector{source, overlay});
                     CHECK(local_parser.get<Simple>() == 4);
                   });
  }
  {
    // Groups + Overlays
    Options::Parser<tmpl::list<Simple, InnerGroupedTag, OuterGroupedTag>>
        parser("");
    const std::string source =
        "Simple: 1\n"
        "OuterGroup:\n"
        "  OuterGroupedTag: 2\n"
        "  InnerGroup:\n"
        "    InnerGroupedTag: 3\n";
    const std::string overlay =
        "OuterGroup:\n"
        "  InnerGroup:\n"
        "    InnerGroupedTag: 5\n";
    parser.parse(source);
    parser.overlay<tmpl::list<InnerGroupedTag>>(overlay);
    check_repeated(parser,
                   [&source, &overlay](const decltype(parser)& local_parser) {
                     CHECK(local_parser.get<Options::Tags::InputSource>() ==
                           std::vector{source, overlay});
                     CHECK(local_parser.get<Simple>() == 1);
                     CHECK(local_parser.get<OuterGroupedTag>() == 2);
                     CHECK(local_parser.get<InnerGroupedTag>() == 5);
                   });
  }
  {
    // Alternatives
    Options::Parser<tmpl::list<AlternativesTag>> parser("");
    const std::string source =
        "AlternativesTag:\n"
        "  A: 1.2\n"
        "  C: 7\n"
        "  F: required\n";
    parser.parse(source);
    check_repeated(parser, [&source](const decltype(parser)& local_parser) {
      CHECK(local_parser.get<Options::Tags::InputSource>() ==
            std::vector{source});
      const auto result = local_parser.get<AlternativesTag>();
      CHECK(result.a_ == 1.2);
      CHECK(result.b_ == -1);
      CHECK(result.c_ == 7);
      CHECK(result.d_.empty());
      CHECK(result.e_ == false);
      CHECK(result.f_ == "required");
    });
  }
}

void test_load_and_check_yaml() {
  INFO("Load and check YAML");
  {
    const auto options = Options::detail::load_and_check_yaml("X: 1", false);
    CHECK(options["X"].as<size_t>() == 1);
  }
  CHECK_THROWS_WITH(
      []() { Options::detail::load_and_check_yaml("X: 1", true); }(),
      Catch::Contains("Missing metadata"));
  {
    const auto options = Options::detail::load_and_check_yaml(
        "---\n"
        "---\n"
        "X: 1\n",
        true);
    CHECK(options["X"].as<size_t>() == 1);
  }
  {
    const auto options = Options::detail::load_and_check_yaml(
        "Executable: RunTests\n"
        "---\n"
        "X: 1\n",
        true);
    CHECK(options["X"].as<size_t>() == 1);
  }
  CHECK_THROWS_WITH(
      []() {
        Options::detail::load_and_check_yaml(
            "Executable: MyExec\n"
            "---\n"
            "X: 1\n",
            true);
      }(),
      Catch::Contains("the running executable is"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Options", "[Unit][Options]") {
  test_options_empty();
  test_options_syntax_error();
  test_options_simple();
  test_options_print_long_help();
  test_options_grouped();
  test_options_suggested();
  test_options_bounded();
  test_options_bounded_vector();
  test_options_array();
  test_options_map();
  test_options_unordered_map();
  test_options_variant();
  test_options_complex_containers();
  test_options_invalid_calls();
  test_options_apply();
  test_options_option_context_default_stream();
  test_options_format();
  test_options_explicit_constructor();
  test_options_input_source();
  test_options_alternatives();
  test_options_overlay();
  test_options_serialization();
  test_load_and_check_yaml();
}
