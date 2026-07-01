// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "PointwiseFunctions/ScalarTensor/DoubleCovariantDerivativeOfScalar.hpp"

namespace {
template <typename Frame, typename DataType>
void test_DDKG_normal_normal_projection(const DataType& used_for_size) {
  Scalar<DataType> (*f)(
      const Scalar<DataType>&, const tnsr::I<DataType, 3, Frame>&,
      const tnsr::II<DataType, 3, Frame>&, const tnsr::i<DataType, 3, Frame>&,
      const tnsr::i<DataType, 3, Frame>&, const Scalar<DataType>&,
      const tnsr::i<DataType, 3, Frame>&) =
      ScalarTensor::DDKG_normal_normal_projection;
  pypp::check_with_random_values<1>(f, "DoubleCovariantDerivativeOfScalar",
                                    "DDKG_normal_normal_projection",
                                    {{{1.0e-2, 0.5}}}, used_for_size);
}

template <typename Frame, typename DataType>
void test_DDKG_normal_spatial_projection(const DataType& used_for_size) {
  tnsr::i<DataType, 3, Frame> (*f)(
      const tnsr::II<DataType, 3, Frame>&, const tnsr::ii<DataType, 3, Frame>&,
      const tnsr::i<DataType, 3, Frame>&, const tnsr::i<DataType, 3, Frame>&) =
      ScalarTensor::DDKG_normal_spatial_projection;
  pypp::check_with_random_values<1>(f, "DoubleCovariantDerivativeOfScalar",
                                    "DDKG_normal_spatial_projection",
                                    {{{1.0e-2, 0.5}}}, used_for_size);
}

template <typename Frame, typename DataType>
void test_DDKG_spatial_spatial_projection(const DataType& used_for_size) {
  tnsr::ii<DataType, 3, Frame> (*f)(
      const tnsr::ii<DataType, 3, Frame>&, const tnsr::Ijj<DataType, 3, Frame>&,
      const Scalar<DataType>&, const tnsr::i<DataType, 3, Frame>&,
      const tnsr::ij<DataType, 3, Frame>&) =
      ScalarTensor::DDKG_spatial_spatial_projection;
  pypp::check_with_random_values<1>(f, "DoubleCovariantDerivativeOfScalar",
                                    "DDKG_spatial_spatial_projection",
                                    {{{1.0e-2, 0.5}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.ScalarTensor.DDScalar",
                  "[Unit][PointwiseFunctions]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/ScalarTensor"};

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_DDKG_normal_normal_projection,
                                    (Frame::Inertial, Frame::Grid))
}
