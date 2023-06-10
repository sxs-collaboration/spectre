// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/DgSubcell/GetTciDecision.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {
SPECTRE_TEST_CASE("Unit.Evolution.Subcell.GetTciDecision",
                  "[Evolution][Unit]") {
  auto box = db::create<db::AddSimpleTags<Tags::TciDecision>>(10);
  CHECK(get_tci_decision(box) == 10);
  db::mutate<Tags::TciDecision>(
      [](const gsl::not_null<int*> tci_decision_ptr) {
        *tci_decision_ptr = -7;
      },
      make_not_null(&box));
  CHECK(get_tci_decision(box) == -7);
}
}  // namespace evolution::dg::subcell
