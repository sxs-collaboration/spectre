// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <utility>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// [class_creation_example]
class CreateFromOptions {
 public:
  struct CfoOption {
    using type = std::string;
    static constexpr Options::String help = {"Option help text"};
  };

  static constexpr Options::String help = {"Class help text"};
  using options = tmpl::list<CfoOption>;

  CreateFromOptions() = default;
  // The Options::Context argument can be left off if unneeded.
  explicit CreateFromOptions(std::string str,
                             const Options::Context& context = {})
      : str_(std::move(str)) {
    if (str_[0] != 'f') {
      PARSE_ERROR(context,
                  "Option must start with an 'f' but is '" << str_ << "'");
    }
  }

  std::string str_{};
};

struct Cfo {
  using type = CreateFromOptions;
  static constexpr Options::String help = {"help"};
};

const char* const input_file_text = R"(
Cfo:
  CfoOption: foo
)";
// [class_creation_example]

// [class_creation_example_with_metavariables]
class CreateFromOptionsWithMetavariables {
 public:
  template <typename Metavariables>
  struct CfoOption {
    static std::string name() noexcept { return Metavariables::option_name(); }
    using type = std::string;
    static constexpr Options::String help = {"Option help text"};
  };

  static constexpr Options::String help = {"Class help text"};
  template <typename Metavariables>
  using options = tmpl::list<CfoOption<Metavariables>>;

  CreateFromOptionsWithMetavariables() = default;
  template <typename Metavariables>
  CreateFromOptionsWithMetavariables(std::string str,
                                     const Options::Context& /*context*/,
                                     Metavariables /*meta*/)
      : str_(std::move(str)),
        expected_(str_ == Metavariables::expected_string()) {}

  std::string str_{};
  bool expected_{false};
};

struct Metavariables {
  static std::string option_name() noexcept { return "MetaName"; }
  static std::string expected_string() noexcept { return "MetaString"; }
};

struct CfoWithMetavariables {
  using type = CreateFromOptionsWithMetavariables;
  static constexpr Options::String help = {"help"};
};

const char* const input_file_text_with_metavariables = R"(
CfoWithMetavariables:
  MetaName: MetaString
)";
// [class_creation_example_with_metavariables]

struct CreateFromOptionsAggregate {
  struct CfoOption {
    using type = std::string;
    static constexpr Options::String help = {"Option help text"};
  };
  static constexpr Options::String help = {"Class help text"};
  using options = tmpl::list<CfoOption>;
  // Define no constructors. The class can be aggregate-initialized.
  std::string str{};
};

struct CfoAggregate {
  using type = CreateFromOptionsAggregate;
  static constexpr Options::String help = {"help"};
};

const char* const input_file_text_aggregate = R"(
CfoAggregate:
  CfoOption: MetaString
)";
}  // namespace

SPECTRE_TEST_CASE("Unit.Options.CustomType", "[Unit][Options]") {
  {
    INFO("Type alias options is not a template");
    Options::Parser<tmpl::list<Cfo>> opts("");
    opts.parse(input_file_text);
    CHECK(opts.get<Cfo>().str_ == "foo");
  }
  {
    INFO("Type alias options is a template");
    Options::Parser<tmpl::list<CfoWithMetavariables>> opts("");
    opts.parse(input_file_text_with_metavariables);
    CHECK(opts.get<CfoWithMetavariables, Metavariables>().str_ == "MetaString");
    CHECK(opts.get<CfoWithMetavariables, Metavariables>().expected_);
  }
  {
    INFO("Aggregate-initialization");
    Options::Parser<tmpl::list<CfoAggregate>> opts("");
    opts.parse(input_file_text_aggregate);
    CHECK(opts.get<CfoAggregate>().str == "MetaString");
  }
}

// [[OutputRegex, In string:.*At line 2 column 3:.Option 'NotOption' is not a
// valid option.]]
SPECTRE_TEST_CASE("Unit.Options.CustomType.error", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<Cfo>> opts("");
  opts.parse("Cfo:\n"
             "  NotOption: foo");
  opts.get<Cfo>();
}

// [[OutputRegex, In string:.*At line 2 column 3:.Option must start with an 'f'
// but is]]
SPECTRE_TEST_CASE("Unit.Options.CustomType.custom_error", "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<Cfo>> opts("");
  opts.parse("Cfo:\n"
             "  CfoOption: zoo");
  opts.get<Cfo>();
}

// [enum_creation_example]
namespace {
enum class CreateFromOptionsAnimal { Cat, Dog };

struct CfoAnimal {
  using type = CreateFromOptionsAnimal;
  static constexpr Options::String help = {"Option help text"};
};
}  // namespace

template <>
struct Options::create_from_yaml<CreateFromOptionsAnimal> {
  template <typename Metavariables>
  static CreateFromOptionsAnimal create(const Options::Option& options) {
    const auto animal = options.parse_as<std::string>();
    if (animal == "Cat") {
      return CreateFromOptionsAnimal::Cat;
    }
    if (animal == "Dog") {
      return CreateFromOptionsAnimal::Dog;
    }
    PARSE_ERROR(options.context(),
                "CreateFromOptionsAnimal must be 'Cat' or 'Dog'");
  }
};
// [enum_creation_example]

SPECTRE_TEST_CASE("Unit.Options.CustomType.specialized", "[Unit][Options]") {
  Options::Parser<tmpl::list<CfoAnimal>> opts("");
  opts.parse("CfoAnimal: Cat");
  CHECK(opts.get<CfoAnimal>() == CreateFromOptionsAnimal::Cat);
}

// [enum_void_creation_header_example]
namespace {
enum class CreateFromOptionsExoticAnimal { MexicanWalkingFish, Platypus };

struct CfoExoticAnimal {
  using type = CreateFromOptionsExoticAnimal;
  static constexpr Options::String help = {"Option help text"};
};
}  // namespace

template <>
struct Options::create_from_yaml<CreateFromOptionsExoticAnimal> {
  template <typename Metavariables>
  static CreateFromOptionsExoticAnimal create(const Options::Option& options) {
    return create<void>(options);
  }
};
template <>
CreateFromOptionsExoticAnimal
Options::create_from_yaml<CreateFromOptionsExoticAnimal>::create<void>(
    const Options::Option& options);
// [enum_void_creation_header_example]

// [enum_void_creation_cpp_example]
template <>
CreateFromOptionsExoticAnimal
Options::create_from_yaml<CreateFromOptionsExoticAnimal>::create<void>(
    const Options::Option& options) {
  const auto animal = options.parse_as<std::string>();
  if (animal == "MexicanWalkingFish") {
    return CreateFromOptionsExoticAnimal::MexicanWalkingFish;
  }
  if (animal == "Platypus") {
    return CreateFromOptionsExoticAnimal::Platypus;
  }
  PARSE_ERROR(options.context(),
              "CreateFromOptionsExoticAnimal must be 'MexicanWalkingFish' or "
              "'Platypus'");
}
// [enum_void_creation_cpp_example]

SPECTRE_TEST_CASE("Unit.Options.CustomType.specialized_void",
                  "[Unit][Options]") {
  const auto helper = [](const std::string& name,
                         const CreateFromOptionsExoticAnimal expected) {
    Options::Parser<tmpl::list<CfoExoticAnimal>> opts("");
    opts.parse("CfoExoticAnimal: " + name);
    CHECK(opts.get<CfoExoticAnimal>() == expected);
  };
  helper("Platypus", CreateFromOptionsExoticAnimal::Platypus);
  helper("MexicanWalkingFish",
         CreateFromOptionsExoticAnimal::MexicanWalkingFish);
}

// [[OutputRegex, In string:.*While parsing option CfoAnimal:.*At line 1
// column 12:.CreateFromOptionsAnimal must be 'Cat' or 'Dog']]
SPECTRE_TEST_CASE("Unit.Options.CustomType.specialized.error",
                  "[Unit][Options]") {
  ERROR_TEST();
  Options::Parser<tmpl::list<CfoAnimal>> opts("");
  opts.parse("CfoAnimal: Mouse");
  opts.get<CfoAnimal>();
}
