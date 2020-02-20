// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

// IWYU pragma: no_include <boost/preprocessor/arithmetic/dec.hpp>
// IWYU pragma: no_include <boost/preprocessor/repetition/enum.hpp>
// IWYU pragma: no_include <boost/preprocessor/tuple/reverse.hpp>

namespace {
template <size_t SpatialDim, typename DataType>
void test_weyl_electric(const DataType& used_for_size) {
  tnsr::ii<DataType, SpatialDim, Frame::Inertial> (*f)(
      const tnsr::ii<DataType, SpatialDim, Frame::Inertial>&,
      const tnsr::ii<DataType, SpatialDim, Frame::Inertial>&,
      const tnsr::II<DataType, SpatialDim, Frame::Inertial>&) =
      &gr::weyl_electric<SpatialDim, Frame::Inertial, DataType>;
  pypp::check_with_random_values<1>(f, "WeylElectric", "weyl_electric_tensor",
                                    {{{-1., 1.}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.WeylElectric",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_weyl_electric, (1, 2, 3));

  TestHelpers::db::test_compute_tag<
      gr::Tags::WeylElectricCompute<3, Frame::Inertial, DataVector>>(
      "WeylElectric");
}
