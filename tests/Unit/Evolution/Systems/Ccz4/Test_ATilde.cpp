// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/ATilde.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t Dim, typename DataType>
void test_compute_a_tilde(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::ii<DataType, Dim, Frame::Inertial> (*)(
          const Scalar<DataType>&,
          const tnsr::ii<DataType, Dim, Frame::Inertial>&,
          const tnsr::ii<DataType, Dim, Frame::Inertial>&,
          const Scalar<DataType>&)>(
          &Ccz4::a_tilde<Dim, Frame::Inertial, DataType>),
      "ATilde", "a_tilde", {{{-1., 1.}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.ATilde", "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env("Evolution/Systems/Ccz4/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_compute_a_tilde, (1, 2, 3));
}
