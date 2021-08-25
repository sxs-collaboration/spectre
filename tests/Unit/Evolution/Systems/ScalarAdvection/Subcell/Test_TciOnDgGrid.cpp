// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/TciOnDgGrid.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace {
// test cases to be covered
enum class TestThis { AllGood, PerssonU };

template <size_t Dim>
void test(const TestThis& test_this) {
  // create DG mesh
  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  // create scalar field U on the DG mesh
  const size_t number_of_points{dg_mesh.number_of_grid_points()};
  Scalar<DataVector> u{number_of_points, 1.0};

  if (test_this == TestThis::PerssonU) {
    // make a troubled cell
    get(u)[number_of_points / 2] += 1.0;
  }

  // check the result
  const double persson_exponent{4.0};
  const bool result = ScalarAdvection::subcell::TciOnDgGrid<Dim>::apply(
      u, dg_mesh, persson_exponent);

  if (test_this == TestThis::AllGood) {
    CHECK_FALSE(result);
  } else {
    CHECK(result);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarAdvection.Subcell.TciOnDgGrid",
                  "[Unit][Evolution]") {
  for (const auto test_this : {TestThis::AllGood, TestThis::PerssonU}) {
    test<1>(test_this);
    test<2>(test_this);
  }
}
