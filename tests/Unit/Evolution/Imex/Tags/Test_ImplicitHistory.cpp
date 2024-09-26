// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Imex/GuessResult.hpp"
#include "Evolution/Imex/Protocols/ImplicitSector.hpp"
#include "Evolution/Imex/Tags/ImplicitHistory.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace {
struct Sector : tt::ConformsTo<imex::protocols::ImplicitSector> {
  using tensors = tmpl::list<>;
  using initial_guess = imex::GuessExplicitResult;

  struct SolveAttempt {
    using tags_from_evolution = tmpl::list<>;
    using simple_tags = tmpl::list<>;
    using compute_tags = tmpl::list<>;
    using source_prep = tmpl::list<>;
    using jacobian_prep = tmpl::list<>;

    struct source : tt::ConformsTo<imex::protocols::ImplicitSource>,
                    tt::ConformsTo<::protocols::StaticReturnApplyable> {
      using return_tags = tmpl::list<>;
      using argument_tags = tmpl::list<>;
      static void apply();
    };

    struct jacobian : tt::ConformsTo<imex::protocols::ImplicitSourceJacobian>,
                      tt::ConformsTo<::protocols::StaticReturnApplyable> {
      using return_tags = tmpl::list<>;
      using argument_tags = tmpl::list<>;
      static void apply();
    };
  };
  using solve_attempts = tmpl::list<SolveAttempt>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Imex.Tags.ImplicitHistory",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<imex::Tags::ImplicitHistory<Sector>>(
      "ImplicitHistory");
}
