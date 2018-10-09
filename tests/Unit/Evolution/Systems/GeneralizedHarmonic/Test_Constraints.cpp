// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

namespace {
template <size_t SpatialDim, typename Frame, typename DataType>
void test_three_index_constraint(const DataType& used_for_size) noexcept {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::iaa<DataType, SpatialDim, Frame> (*)(
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&)>(
          &GeneralizedHarmonic::three_index_constraint<SpatialDim, Frame,
                                                       DataType>),
      "numpy", "subtract", {{{-10.0, 10.0}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.ThreeIndexConstraint",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GeneralizedHarmonic/"};

  test_three_index_constraint<1, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<1, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<1, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_three_index_constraint<1, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());

  test_three_index_constraint<2, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<2, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<2, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_three_index_constraint<2, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());

  test_three_index_constraint<3, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<3, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<3, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_three_index_constraint<3, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());
}
