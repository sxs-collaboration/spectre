// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::Tags::InternalFace {
namespace {
template <size_t Dim>
void test() {
  TestHelpers::db::test_simple_tag<MagnitudeOfNormal>("MagnitudeOfNormal");
  TestHelpers::db::test_simple_tag<NormalCovector<Dim>>("NormalCovector");
  TestHelpers::db::test_simple_tag<NormalCovectorAndMagnitude<Dim>>(
      "NormalCovectorAndMagnitude");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.InternalFaceTags", "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace evolution::dg::Tags::InternalFace
