// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/DoubleCovariantDerivativeOfCoupling.hpp"

namespace {
template <typename DataType>
void test_DDCoupling_normal_normal_projection(const DataType& used_for_size) {
  Scalar<DataType> (*f)(const Scalar<DataType>&, const Scalar<DataType>&,
                        const Scalar<DataType>&, const Scalar<DataType>&) =
      &ScalarTensor::sgb::DDCoupling_normal_normal_projection;
  pypp::check_with_random_values<1>(f, "DoubleCovariantDerivativeOfCoupling",
                                    "DDCoupling_normal_normal_projection",
                                    {{{-1., 1.}}}, used_for_size);
}

template <typename Frame, typename DataType>
void test_DDCoupling_normal_spatial_projection(const DataType& used_for_size) {
  tnsr::i<DataType, 3, Frame> (*f)(
      const Scalar<DataType>&, const Scalar<DataType>&, const Scalar<DataType>&,
      const tnsr::i<DataType, 3, Frame>&, const tnsr::i<DataType, 3, Frame>&) =
      &ScalarTensor::sgb::DDCoupling_normal_spatial_projection;
  pypp::check_with_random_values<1>(f, "DoubleCovariantDerivativeOfCoupling",
                                    "DDCoupling_normal_spatial_projection",
                                    {{{-1., 1.}}}, used_for_size);
}

template <typename Frame, typename DataType>
void test_DDCoupling_spatial_spatial_projection(const DataType& used_for_size) {
  tnsr::ii<DataType, 3, Frame> (*f)(
      const Scalar<DataType>&, const Scalar<DataType>&,
      const tnsr::i<DataType, 3, Frame>&, const tnsr::ii<DataType, 3, Frame>&) =
      &ScalarTensor::sgb::DDCoupling_spatial_spatial_projection;
  pypp::check_with_random_values<1>(f, "DoubleCovariantDerivativeOfCoupling",
                                    "DDCoupling_spatial_spatial_projection",
                                    {{{-1., 1.}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.ScalarTensor.sgb.DDCoupling",
                  "[Unit][PointwiseFunctions]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet"};

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  test_DDCoupling_normal_normal_projection(d);
  test_DDCoupling_normal_normal_projection(dv);

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_DDCoupling_normal_spatial_projection,
                                    (Frame::Inertial, Frame::Grid));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_DDCoupling_spatial_spatial_projection,
                                    (Frame::Inertial, Frame::Grid));
}
