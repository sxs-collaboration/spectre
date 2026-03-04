// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/CurvedScalarWave/ApplyTensorYlmFilter.hpp"
#include "Helpers/NumericalAlgorithms/SphericalHarmonics/Test_ApplyTensorYlmFilter.hpp"

namespace CurvedScalarWave {

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.CurvedScalarWave.ApplyTensorYlmFilter",
    "[NumericalAlgorithms][Unit]") {
  const auto apply_filter =
      [](const auto vars_nodal, const auto vars_storage,
         const auto& jac_inertial_to_grid, const auto& jac_grid_to_inertial,
         const auto& filter_matrices, const size_t ell_max,
         const size_t radial_extents) {
        apply_tensor_ylm_filter(vars_nodal, vars_storage, jac_inertial_to_grid,
                                jac_grid_to_inertial, filter_matrices.scalar,
                                filter_matrices.i, ell_max, radial_extents);
      };
  ylm::TensorYlm::test_apply_filter<
      filter_detail::sw_vars_list<Frame::Inertial>, false>(0, apply_filter);
  ylm::TensorYlm::test_apply_filter<
      filter_detail::sw_vars_list<Frame::Inertial>, false>(5, apply_filter);
}
}  // namespace CurvedScalarWave
