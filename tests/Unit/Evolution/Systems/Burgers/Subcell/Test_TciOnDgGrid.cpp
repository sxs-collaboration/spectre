// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Burgers/Subcell/TciOnDgGrid.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Burgers.Subcell.TciOnDgGrid",
                  "[Unit][Evolution]") {
  enum class TestThis { AllGood, PerssonU };

  for (const auto test_this : {TestThis::AllGood, TestThis::PerssonU}) {
    const Mesh<1> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
    const size_t number_of_points{dg_mesh.number_of_grid_points()};

    Scalar<DataVector> u{number_of_points, 1.0};

    if (test_this == TestThis::PerssonU) {
      // make a troubled cell
      get(u)[number_of_points / 2] += 1.0;
    }

    // check the result
    const double persson_exponent{4.0};
    const bool result =
        Burgers::subcell::TciOnDgGrid::apply(u, dg_mesh, persson_exponent);
    if (test_this == TestThis::AllGood) {
      CHECK_FALSE(result);
    } else {
      CHECK(result);
    }
  }
}
