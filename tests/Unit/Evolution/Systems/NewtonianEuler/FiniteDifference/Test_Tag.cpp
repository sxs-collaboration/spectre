// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Tag.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
template <size_t Dim>
void test() {
  TestHelpers::db::test_simple_tag<
      NewtonianEuler::fd::Tags::Reconstructor<Dim>>("Reconstructor");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Fd.Tag",
                  "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
