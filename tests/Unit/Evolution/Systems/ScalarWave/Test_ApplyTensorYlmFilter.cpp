// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ScalarWave/ApplyTensorYlmFilter.hpp"
#include "Helpers/NumericalAlgorithms/TensorYlm/Test_ApplyTensorYlmFilter.hpp"

namespace ylm::TensorYlm {

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarWave.ApplyTensorYlmFilter",
                  "[NumericalAlgorithms][Unit]") {
  test_apply_filter<filter_detail::sw_vars_list>(0);
  test_apply_filter<filter_detail::sw_vars_list>(5);
}
}  // namespace ylm::TensorYlm
