// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/NewtonianMhd/FiniteDifference/Tag.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
template <size_t Dim>
void test() {
  TestHelpers::db::test_simple_tag<NewtonianMhd::fd::Tags::Reconstructor<Dim>>(
      "Reconstructor");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianMhd.Fd.Tag",
                  "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
