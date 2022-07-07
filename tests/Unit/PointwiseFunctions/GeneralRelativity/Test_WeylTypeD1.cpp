// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylTypeD1.hpp"

// IWYU pragma: no_include <boost/preprocessor/arithmetic/dec.hpp>
// IWYU pragma: no_include <boost/preprocessor/repetition/enum.hpp>
// IWYU pragma: no_include <boost/preprocessor/tuple/reverse.hpp>

namespace {
template <size_t SpatialDim, typename DataType>
void test_weyl_type_D1(const DataType& used_for_size) {
  tnsr::ii<DataType, SpatialDim, Frame::Inertial> (*f)(
      const tnsr::ii<DataType, SpatialDim, Frame::Inertial>&,
      const tnsr::ii<DataType, SpatialDim, Frame::Inertial>&,
      const tnsr::II<DataType, SpatialDim, Frame::Inertial>&) =
      &gr::weyl_type_D1_tensor<SpatialDim, Frame::Inertial, DataType>;
  pypp::check_with_random_values<1>(f, "WeylTypeD1", "weyl_TypeD1_tensor",
                                    {{{-1., 1.}}}, used_for_size);
}

template <size_t SpatialDim, typename DataType>
void test_weyl_type_D1_scalar(const DataType& used_for_size) {
  Scalar<DataType> (*f)(
      const tnsr::ii<DataType, SpatialDim, Frame::Inertial>&,
      const tnsr::II<DataType, SpatialDim, Frame::Inertial>&) =
      &gr::weyl_type_D1_scalar<SpatialDim, Frame::Inertial, DataType>;
  pypp::check_with_random_values<1>(f, "WeylTypeD1Scalar",
                                    "weyl_type_D1_scalar", {{{-1., 1.}}},
                                    used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.WeylTypeD1",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_weyl_type_D1, (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_weyl_type_D1_scalar, (1, 2, 3));
  //test_compute_item_in_databox<3>(d);
  //test_compute_item_in_databox<3>(dv);
}
