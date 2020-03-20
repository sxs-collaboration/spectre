// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.DiscontinuousGalerkin.Formulation",
                  "[Unit][NumericalAlgorithms]") {
  CHECK(get_output(dg::Formulation::StrongInertial) == "StrongInertial");
  CHECK(get_output(dg::Formulation::WeakInertial) == "WeakInertial");
}
