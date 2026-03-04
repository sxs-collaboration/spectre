// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/CurvedScalarWave/ApplyTensorYlmFilter.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/SphericalHarmonics/Test_ApplyTensorYlmFilter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filter.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave {
namespace {
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Filters::Filter, tmpl::list<TensorYlmFilter>>>;
  };
};
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.CurvedScalarWave.ApplyTensorYlmFilter",
    "[NumericalAlgorithms][Unit]") {
  register_factory_classes_with_charm<Metavariables>();

  const auto created_filter = TestHelpers::test_creation<
      std::unique_ptr<Filters::Filter>, Metavariables>(
      "TensorYlmFilter:\n"
      "  NumModesToKill: 2\n"
      "  HalfPower: 5");
  const auto& concrete_filter =
      dynamic_cast<const TensorYlmFilter&>(*created_filter);
  CHECK(concrete_filter == TensorYlmFilter{2, 5});
  CHECK(concrete_filter.blocks_to_filter() == std::nullopt);

  const auto deserialized_filter = serialize_and_deserialize(created_filter);
  CHECK(dynamic_cast<const TensorYlmFilter&>(*deserialized_filter) ==
        concrete_filter);

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
