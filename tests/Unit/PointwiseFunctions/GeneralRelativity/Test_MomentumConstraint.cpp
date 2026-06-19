// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "PointwiseFunctions/GeneralRelativity/MomentumConstraint.hpp"

namespace {
template <size_t SpatialDim, typename Frame, typename DataType>
void test_momentum_constraint_in_vacuum(const DataType& used_for_size) {
  tnsr::i<DataType, SpatialDim, Frame> (*f)(
      const tnsr::ijj<DataType, SpatialDim, Frame>&,
      const tnsr::i<DataType, SpatialDim, Frame>&,
      const tnsr::II<DataType, SpatialDim, Frame>&) =
      &gr::momentum_constraint_in_vacuum<DataType, SpatialDim, Frame>;
  pypp::check_with_random_values<1>(f, "MomentumConstraint",
                                    "momentum_constraint_in_vacuum",
                                    {{{-1., 1.}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.GeneralRelativity.MomentumConstraint",
    "[Unit][PointwiseFunctions]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/GeneralRelativity"};

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_momentum_constraint_in_vacuum,
                                    (1, 2, 3), (Frame::Inertial, Frame::Grid));
}
