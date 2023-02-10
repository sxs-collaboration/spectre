// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "PointwiseFunctions/Hydro/ComovingMagneticField.hpp"

namespace hydro {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.ComovingMagneticField",
                  "[Unit][Hydro]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/Hydro/");
  const DataVector used_for_size(5);
  pypp::check_with_random_values<1>(
      static_cast<tnsr::a<DataVector, 3> (*)(
          const tnsr::i<DataVector, 3>&, const tnsr::i<DataVector, 3>&,
          const Scalar<DataVector>&, const Scalar<DataVector>&,
          const tnsr::I<DataVector, 3>&, const Scalar<DataVector>&)>(
          &comoving_magnetic_field_one_form<DataVector>),
      "TestFunctions", "comoving_magnetic_field_one_form", {{{0.0, 1.0}}},
      used_for_size);
  pypp::check_with_random_values<1>(
      static_cast<Scalar<DataVector> (*)(const Scalar<DataVector>&,
                                         const Scalar<DataVector>&,
                                         const Scalar<DataVector>&)>(
          &comoving_magnetic_field_squared<DataVector>),
      "TestFunctions", "comoving_magnetic_field_squared", {{{0.0, 1.0}}},
      used_for_size);
}

}  // namespace hydro
