// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Imex/GuessResult.hpp"
#include "Utilities/Gsl.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Imex.GuessResult", "[Unit][Evolution]") {
  // Since the mutators don't have information about the implicit
  // sector, they can't actually use anything in the DataBox.
  auto box = db::create<db::AddSimpleTags<>>();

  // Technically, the homogeneous terms should be a Variables, but
  // there's nothing a mutator with no return tags can possibly do
  // with them, so if it compiles with a dummy type it will work in
  // real use.
  struct Dummy {};
  const std::vector<imex::GuessResult> result =
      db::mutate_apply<imex::GuessExplicitResult>(make_not_null(&box), Dummy{},
                                                  2.0);
  CHECK(result.empty());

  CHECK_THROWS_WITH(
      db::mutate_apply<imex::NoJacobianBecauseSolutionIsAnalytic>(
          make_not_null(&box)),
      Catch::Matchers::ContainsSubstring(
          "initial guess did not return GuessResult::ExactSolution"));
}
