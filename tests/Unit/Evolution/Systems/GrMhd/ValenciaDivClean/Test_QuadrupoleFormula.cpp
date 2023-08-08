// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/QuadrupoleFormula.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.QuadrupoleFormula",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_compute_tag<
      grmhd::ValenciaDivClean::Tags::QuadrupoleMomentCompute<
               DataVector, 3, Frame::Inertial>>("QuadrupoleMoment");
  TestHelpers::db::test_compute_tag<
      grmhd::ValenciaDivClean::Tags::QuadrupoleMomentDerivativeCompute<
               DataVector, 3, Frame::Inertial>>("QuadrupoleMomentDerivative");
}
