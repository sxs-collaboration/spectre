// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/OptionTags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Utilities/TypeTraits.hpp"

namespace domain {

template <bool Override>
struct Metavariables {
  static constexpr size_t volume_dim = 3;
  static constexpr bool override_cubic_functions_of_time = Override;
};

SPECTRE_TEST_CASE("Unit.Domain.FunctionsOfTime.Tags", "[Domain][Unit]") {
  TestHelpers::db::test_simple_tag<Tags::FunctionsOfTime>("FunctionsOfTime");

  CHECK(std::is_same_v<
        Tags::FunctionsOfTime::option_tags<Metavariables<true>>,
        tmpl::list<
            domain::OptionTags::DomainCreator<Metavariables<true>::volume_dim>,
            domain::FunctionsOfTime::OptionTags::FunctionOfTimeFile,
            domain::FunctionsOfTime::OptionTags::FunctionOfTimeNameMap>>);

  CHECK(std::is_same_v<Tags::FunctionsOfTime::option_tags<Metavariables<false>>,
                       tmpl::list<domain::OptionTags::DomainCreator<
                           Metavariables<false>::volume_dim>>>);
}
}  // namespace domain
