// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/ScalarGaussBonnet/Tags.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/ScalarGaussBonnet/ScalarMomentum.hpp"

namespace sgb {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.ScalarGaussBonnet.ScalarMomentum",
                  "[Unit][PointwiseFunctions]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/ScalarGaussBonnet"};
  const DataVector used_for_size{5};
  pypp::check_with_random_values<1>(
      static_cast<void (*)(gsl::not_null<Scalar<DataVector>*>,
                           const tnsr::i<DataVector, 3, Frame::Inertial>&,
                           const tnsr::I<DataVector, 3>&,
                           const Scalar<DataVector>&)>(&scalar_momentum),
      "ScalarMomentum", {"compute_pi"}, {{{0., 1.}}}, used_for_size);

  TestHelpers::db::test_compute_tag<Tags::PiCompute<
      gr::Tags::Shift<DataVector, 3>, CurvedScalarWave::Tags::Pi>>("Pi");
  TestHelpers::db::test_compute_tag<Tags::PiCompute<
      sgb::Tags::RolledOffShift, sgb::Tags::PiWithRolledOffShift>>(
      "PiWithRolledOffShift");
}

}  // namespace sgb
