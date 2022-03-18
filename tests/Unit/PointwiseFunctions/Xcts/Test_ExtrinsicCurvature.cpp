// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "PointwiseFunctions/Xcts/ExtrinsicCurvature.hpp"

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Xcts.ExtrinsicCurvature",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{"PointwiseFunctions/Xcts"};
  const DataVector used_for_size{5};
  pypp::check_with_random_values<1>(
      static_cast<void (*)(
          gsl::not_null<tnsr::ii<DataVector, 3>*>, const Scalar<DataVector>&,
          const Scalar<DataVector>&, const tnsr::ii<DataVector, 3>&,
          const tnsr::II<DataVector, 3>&, const Scalar<DataVector>&)>(
          &Xcts::extrinsic_curvature<DataVector>),
      "ExtrinsicCurvature", {"extrinsic_curvature"}, {{{-1., 1.}}},
      used_for_size);
}
