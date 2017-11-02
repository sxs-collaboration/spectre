// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <catch.hpp>
#include <string>
#include <utility>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
/// [class_creation_example]
template <typename>
class CreateFromOptions;
struct CFO {
  using type = CreateFromOptions<int>;
  static constexpr OptionString_t help = {"help"};
};

template <typename T>
class CreateFromOptions {
 public:
  struct Option {
    using type = std::string;
    static constexpr OptionString_t help = {"Option help text"};
  };
  using options = tmpl::list<Option>;
  static constexpr OptionString_t help = {"Class help text"};

  CreateFromOptions() = default;
  explicit CreateFromOptions(std::string str) : str_(std::move(str)) {}

  std::string str_;
};
/// [class_creation_example]
}  // namespace

SPECTRE_TEST_CASE("Unit.Options.CustomType", "[Unit][Options]") {
  Options<tmpl::list<CFO>> opts("");
  opts.parse("CFO:\n"
             "  Option: foo");
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

/// [enum_creation_example]
namespace {
enum class CreateFromOptionsAnimal { Cat, Dog };

struct CFOAnimal {
  using type = CreateFromOptionsAnimal;
  static constexpr OptionString_t help = {"Option help text"};
};
}  // namespace

template <>
struct create_from_yaml<CreateFromOptionsAnimal> {
  static CreateFromOptionsAnimal create(const Option_t& options) {
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
