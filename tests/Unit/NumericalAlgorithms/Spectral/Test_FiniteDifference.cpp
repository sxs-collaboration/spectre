// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GetOutput.hpp"

namespace Spectral {
SPECTRE_TEST_CASE("Unit.Numerical.Spectral.Fd.Points",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  CHECK(
      DataVector{-2.0 / 3.0, 0.0, 2.0 / 3.0} ==
      collocation_points<Basis::FiniteDifference, Quadrature::CellCentered>(3));
  CHECK(
      DataVector{-0.75, -0.25, 0.25, 0.75} ==
      collocation_points<Basis::FiniteDifference, Quadrature::CellCentered>(4));
  CHECK(get_output(Basis::FiniteDifference) == "FiniteDifference");
  CHECK(get_output(Quadrature::CellCentered) == "CellCentered");
}
}  // namespace Spectral
