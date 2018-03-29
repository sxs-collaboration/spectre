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
  // The OptionContext argument can be left off if unneeded.
  CreateFromOptions(std::string str, const OptionContext& context)
      : str_(std::move(str)) {
    if (str_[0] != 'f') {
      PARSE_ERROR(context, "Option must start with an f");
    }
  }

  std::string str_;
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
  CHECK(opts.get<CFO>().str_ == "foo");
}

// [[OutputRegex, In string:.*At line 2 column 3:.Option 'NotOption' is not a
// valid option.]]
SPECTRE_TEST_CASE("Unit.Options.CustomType.error", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<CFO>> opts("");
  opts.parse("CFO:\n"
             "  NotOption: foo");
  opts.get<CFO>();
}

// [[OutputRegex, In string:.*At line 2 column 3:.Option must start with an f]]
SPECTRE_TEST_CASE("Unit.Options.CustomType.custom_error", "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<CFO>> opts("");
  opts.parse("CFO:\n"
             "  Option: zoo");
  opts.get<CFO>();
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

// [[OutputRegex, In string:.*While parsing option CFOAnimal:.*At line 1
// column 12:.CreateFromOptionsAnimal must be 'Cat' or 'Dog']]
SPECTRE_TEST_CASE("Unit.Options.CustomType.specialized.error",
                  "[Unit][Options]") {
  ERROR_TEST();
  Options<tmpl::list<CFOAnimal>> opts("");
  opts.parse("CFOAnimal: Mouse");
  opts.get<CFOAnimal>();
}
