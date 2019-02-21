// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>
#include <utility>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/TMPL.hpp"

namespace {
/// [class_creation_example]
template <bool Valid>
struct Metavariables {
  static constexpr bool valid = Valid;
};

template <typename>
class CreateFromOptions;
struct CFO {
  using type = CreateFromOptions<int>;
  static constexpr OptionString help = {"help"};
};

template <typename T>
class CreateFromOptions {
 public:
  struct Option {
    using type = std::string;
    static constexpr OptionString help = {"Option help text"};
  };
  using options = tmpl::list<Option>;
  static constexpr OptionString help = {"Class help text"};

  CreateFromOptions() = default;
  // The Metavariables arguments can be left off if unneeded, and the
  // OptionContext as well.
  template <typename Metavariables>
  CreateFromOptions(std::string str, const OptionContext& context,
                    Metavariables /*meta*/)
      : str_(std::move(str)), valid_(Metavariables::valid) {
    if (str_[0] != 'f') {
      PARSE_ERROR(context,
                  "Option must start with an 'f' but is '" << str_ << "'");
    }
  }

  std::string str_{};
  bool valid_{false};
};

const char* const input_file_text = R"(
CFO:
  Option: foo
)";
/// [class_creation_example]
}  // namespace

SPECTRE_TEST_CASE("Unit.Options.CustomType", "[Unit][Options]") {
  Options<tmpl::list<CFO>> opts("");
  opts.parse(input_file_text);
  CHECK(opts.get<CFO, Metavariables<true>>().str_ == "foo");
  CHECK(opts.get<CFO, Metavariables<true>>().valid_);
  CHECK_FALSE(opts.get<CFO, Metavariables<false>>().valid_);
}

// [[OutputRegex, In string:.*At line 2 column 3:.Option 'NotOption' is not a
// valid option.]]
SPECTRE_TEST_CASE("Unit.Options.CustomType.error", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<CFO>> opts("");
  opts.parse("CFO:\n"
             "  NotOption: foo");
  opts.get<CFO, Metavariables<true>>();
}

// [[OutputRegex, In string:.*At line 2 column 3:.Option must start with an 'f'
// but is]]
SPECTRE_TEST_CASE("Unit.Options.CustomType.custom_error", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<CFO>> opts("");
  opts.parse("CFO:\n"
             "  Option: zoo");
  opts.get<CFO, Metavariables<true>>();
}

/// [enum_creation_example]
namespace {
enum class CreateFromOptionsAnimal { Cat, Dog };

struct CFOAnimal {
  using type = CreateFromOptionsAnimal;
  static constexpr OptionString help = {"Option help text"};
};
}  // namespace

template <>
struct create_from_yaml<CreateFromOptionsAnimal> {
  template <typename Metavariables>
  static CreateFromOptionsAnimal create(const Option& options) {
    const std::string animal = options.parse_as<std::string>();
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
/// [enum_creation_example]

SPECTRE_TEST_CASE("Unit.Options.CustomType.specialized", "[Unit][Options]") {
  Options<tmpl::list<CFOAnimal>> opts("");
  opts.parse("CFOAnimal: Cat");
  CHECK(opts.get<CFOAnimal>() == CreateFromOptionsAnimal::Cat);
}

/// [enum_void_creation_header_example]
namespace {
enum class CreateFromOptionsExoticAnimal { MexicanWalkingFish, Platypus };

struct CFOExoticAnimal {
  using type = CreateFromOptionsExoticAnimal;
  static constexpr OptionString help = {"Option help text"};
};
}  // namespace

template <>
struct create_from_yaml<CreateFromOptionsExoticAnimal> {
  template <typename Metavariables>
  static CreateFromOptionsExoticAnimal create(const Option& options) {
    return create<void>(options);
  }
};
template <>
CreateFromOptionsExoticAnimal
create_from_yaml<CreateFromOptionsExoticAnimal>::create<void>(
    const Option& options);
/// [enum_void_creation_header_example]

/// [enum_void_creation_cpp_example]
template <>
CreateFromOptionsExoticAnimal
create_from_yaml<CreateFromOptionsExoticAnimal>::create<void>(
    const Option& options) {
  const std::string animal = options.parse_as<std::string>();
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
/// [enum_void_creation_cpp_example]

SPECTRE_TEST_CASE("Unit.Options.CustomType.specialized_void",
                  "[Unit][Options]") {
  const auto helper = [](const std::string& name,
                         const CreateFromOptionsExoticAnimal expected) {
    Options<tmpl::list<CFOExoticAnimal>> opts("");
    opts.parse("CFOExoticAnimal: " + name);
    CHECK(opts.get<CFOExoticAnimal>() == expected);
  };
  helper("Platypus", CreateFromOptionsExoticAnimal::Platypus);
  helper("MexicanWalkingFish",
         CreateFromOptionsExoticAnimal::MexicanWalkingFish);
}

// [[OutputRegex, In string:.*While parsing option CFOAnimal:.*At line 1
// column 12:.CreateFromOptionsAnimal must be 'Cat' or 'Dog']]
SPECTRE_TEST_CASE("Unit.Options.CustomType.specialized.error",
                  "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<CFOAnimal>> opts("");
  opts.parse("CFOAnimal: Mouse");
  opts.get<CFOAnimal>();
}
