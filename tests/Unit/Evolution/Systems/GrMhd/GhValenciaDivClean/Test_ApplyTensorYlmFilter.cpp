// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/ApplyTensorYlmFilter.hpp"
#include "Helpers/NumericalAlgorithms/TensorYlm/Test_ApplyTensorYlmFilter.hpp"

namespace ylm::TensorYlm {

// [[TimeOut, 30]]
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GrMhd.GhValenciaDivClean.ApplyTensorYlmFilter",
    "[NumericalAlgorithms][Unit]") {
  using vars_list = grmhd::GhValenciaDivClean::filter_detail::ghmhd_vars_list;
  test_apply_filter<vars_list>(0);
  test_apply_filter<vars_list>(5);
}
}  // namespace ylm::TensorYlm
