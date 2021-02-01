// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "Elliptic/DiscontinuousGalerkin/Penalty.hpp"

SPECTRE_TEST_CASE("Unit.Elliptic.DG.Penalty", "[Unit][Elliptic]") {
  const size_t num_points = 3;
  const DataVector element_size{1., 2., 3.};
  const double penalty_parameter = 1.5;
  const DataVector expected_penalty{13.5, 6.75, 4.5};
  CHECK_ITERABLE_APPROX(
      elliptic::dg::penalty(element_size, num_points, penalty_parameter),
      expected_penalty);
}
