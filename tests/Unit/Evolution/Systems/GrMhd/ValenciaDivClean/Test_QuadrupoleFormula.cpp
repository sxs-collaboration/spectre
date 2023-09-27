// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/QuadrupoleFormula.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/Hydro/TransportVelocity.hpp"

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.QuadrupoleFormula",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_compute_tag<
      grmhd::ValenciaDivClean::Tags::QuadrupoleMomentCompute<
               DataVector, 3, Frame::Inertial>>("QuadrupoleMoment");
  TestHelpers::db::test_compute_tag<
      grmhd::ValenciaDivClean::Tags::QuadrupoleMomentDerivativeCompute<
          DataVector, 3,
          ::Events::Tags::ObserverCoordinates<3, Frame::Inertial>,
          hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>,
          Frame::Inertial>>("QuadrupoleMomentDerivative(SpatialVelocity)");
  TestHelpers::db::test_compute_tag<
      grmhd::ValenciaDivClean::Tags::QuadrupoleMomentDerivativeCompute<
          DataVector, 3,
          ::Events::Tags::ObserverCoordinates<3, Frame::Inertial>,
          hydro::Tags::TransportVelocity<DataVector, 3, Frame::Inertial>,
          Frame::Inertial>>("QuadrupoleMomentDerivative(TransportVelocity)");
}
