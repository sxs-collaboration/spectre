// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/TensorYlm/Helpers.hpp"

namespace {
void test_component_spin_weight() {
  CHECK(ylm::TensorYlm::helpers::component_spin_weight<
            Scalar<DataVector>::structure>(0) == 0);

  CHECK(ylm::TensorYlm::helpers::component_spin_weight<
            tnsr::i<DataVector, 3, Frame::Grid>::structure>(0) == 0);
  CHECK(ylm::TensorYlm::helpers::component_spin_weight<
            tnsr::i<DataVector, 3, Frame::Grid>::structure>(1) == -1);
  CHECK(ylm::TensorYlm::helpers::component_spin_weight<
            tnsr::i<DataVector, 3, Frame::Grid>::structure>(2) == 1);

  CHECK(ylm::TensorYlm::helpers::component_spin_weight<
            tnsr::ii<DataVector, 3, Frame::Grid>::structure>(0) == 0);
  CHECK(ylm::TensorYlm::helpers::component_spin_weight<
            tnsr::ii<DataVector, 3, Frame::Grid>::structure>(1) == -1);
  CHECK(ylm::TensorYlm::helpers::component_spin_weight<
            tnsr::ii<DataVector, 3, Frame::Grid>::structure>(2) == 1);
  CHECK(ylm::TensorYlm::helpers::component_spin_weight<
            tnsr::ii<DataVector, 3, Frame::Grid>::structure>(3) == -2);
  CHECK(ylm::TensorYlm::helpers::component_spin_weight<
            tnsr::ii<DataVector, 3, Frame::Grid>::structure>(4) == 0);
  CHECK(ylm::TensorYlm::helpers::component_spin_weight<
            tnsr::ii<DataVector, 3, Frame::Grid>::structure>(5) == 2);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.TensorYlm.Helpers", "[NumericalAlgorithms][Unit]") {
  test_component_spin_weight();
}
