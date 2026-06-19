// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "PointwiseFunctions/GeneralRelativity/HamiltonianConstraint.hpp"

namespace {
template <size_t SpatialDim, typename Frame, typename DataType>
void test_hamiltonian_constraint_in_vacuum(const DataType& used_for_size) {
  Scalar<DataType> (*f)(const Scalar<DataType>&, const Scalar<DataType>&,
                        const tnsr::II<DataType, SpatialDim, Frame>&,
                        const tnsr::ii<DataType, SpatialDim, Frame>&) =
      &gr::hamiltonian_constraint_in_vacuum<DataType, SpatialDim, Frame>;
  pypp::check_with_random_values<1>(f, "HamiltonianConstraint",
                                    "hamiltonian_constraint_in_vacuum",
                                    {{{-1., 1.}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.GeneralRelativity.HamiltonianConstraint",
    "[Unit][PointwiseFunctions]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/GeneralRelativity"};

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_hamiltonian_constraint_in_vacuum,
                                    (1, 2, 3), (Frame::Inertial, Frame::Grid));
}
