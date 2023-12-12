// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeDerivativeOfGothG.hpp"

namespace gr {

template <typename DataType, size_t SpatialDim>
void test_compute_spacetime_deriv_of_goth_g(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::aBB<DataVector, SpatialDim, Frame::Inertial> (*)(
          const tnsr::AA<DataType, SpatialDim, Frame::Inertial>&,
          const tnsr::abb<DataType, SpatialDim, Frame::Inertial>&,
          const Scalar<DataType>&,
          const Scalar<DataType>&,
          const tnsr::i<DataType, SpatialDim, Frame::Inertial>&,
          const Scalar<DataType>&,
          const tnsr::a<DataType, SpatialDim, Frame::Inertial>&)>(
          &spacetime_deriv_of_goth_g<DataType, SpatialDim, Frame::Inertial>),
      "SpacetimeDerivativeOfGothG", "spacetime_deriv_of_goth_g",
      {{{0.0, 1.0}}}, used_for_size);
}

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.GeneralRelativity.SpacetimeDerivativeOfGothG",
    "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");
  const DataVector used_for_size(5);
  test_compute_spacetime_deriv_of_goth_g<DataVector, 1>(used_for_size);
  test_compute_spacetime_deriv_of_goth_g<DataVector, 3>(used_for_size);
}

}  // namespace gr
