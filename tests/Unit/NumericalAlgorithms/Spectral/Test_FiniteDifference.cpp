// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Basis.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GetOutput.hpp"

namespace Spectral {
SPECTRE_TEST_CASE("Unit.Numerical.Spectral.Fd.Points",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  CHECK_ITERABLE_APPROX(
      SINGLE_ARG(DataVector{-2.0 / 3.0, 0.0, 2.0 / 3.0}),
      SINGLE_ARG(
          collocation_points<SpatialDiscretization::Basis::FiniteDifference,
                             SpatialDiscretization::Quadrature::CellCentered>(
              3)));
  CHECK(DataVector{-0.75, -0.25, 0.25, 0.75} ==
        collocation_points<SpatialDiscretization::Basis::FiniteDifference,
                           SpatialDiscretization::Quadrature::CellCentered>(4));
  CHECK(get_output(SpatialDiscretization::Basis::FiniteDifference) ==
        "FiniteDifference");
  CHECK(get_output(SpatialDiscretization::Quadrature::CellCentered) ==
        "CellCentered");

  CHECK(DataVector{-1.0, 0.0, 1.0} ==
        collocation_points<SpatialDiscretization::Basis::FiniteDifference,
                           SpatialDiscretization::Quadrature::FaceCentered>(3));
  CHECK_ITERABLE_APPROX(
      SINGLE_ARG(DataVector{-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0}),
      SINGLE_ARG(
          collocation_points<SpatialDiscretization::Basis::FiniteDifference,
                             SpatialDiscretization::Quadrature::FaceCentered>(
              4)));
  CHECK(get_output(SpatialDiscretization::Quadrature::FaceCentered) ==
        "FaceCentered");
}
}  // namespace Spectral
