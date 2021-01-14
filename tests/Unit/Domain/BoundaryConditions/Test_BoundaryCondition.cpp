// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <pup.h>
#include <string>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Parallel/CharmPupable.hpp"

namespace {
namespace helpers = TestHelpers::domain::BoundaryConditions;

template <size_t Dim>
void test() {
  const Direction<Dim> expected_direction = Direction<Dim>::upper_xi();
  const std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      bc_default = std::make_unique<helpers::TestBoundaryCondition<Dim>>(
          expected_direction);
  const auto bc_default_deserialized = serialize_and_deserialize(bc_default);

  const std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> bc =
      std::make_unique<helpers::TestBoundaryCondition<Dim>>(expected_direction,
                                                            10);
  const auto bc_deserialized = serialize_and_deserialize(bc);

  const auto perform_checks =
      [&expected_direction](
          const std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>&
              boundary_condition,
          const size_t expected_block_id) {
        const auto& derived_boundary_condition =
            dynamic_cast<const helpers::TestBoundaryCondition<Dim>&>(
                *boundary_condition);
        CHECK(derived_boundary_condition.direction() == expected_direction);
        CHECK(derived_boundary_condition.block_id() == expected_block_id);
        CHECK(derived_boundary_condition ==
              helpers::TestBoundaryCondition<Dim>{expected_direction,
                                                  expected_block_id});
        CHECK(derived_boundary_condition !=
              helpers::TestBoundaryCondition<Dim>{expected_direction.opposite(),
                                                  expected_block_id});
        CHECK(derived_boundary_condition !=
              helpers::TestBoundaryCondition<Dim>{expected_direction,
                                                  expected_block_id + 1});
      };
  perform_checks(bc_default, 0);
  perform_checks(bc_default_deserialized, 0);
  perform_checks(bc, 10);
  perform_checks(bc_deserialized, 10);
}

SPECTRE_TEST_CASE("Unit.Domain.BoundaryConditions.BoundaryCondition",
                  "[Unit][Domain]") {
  // This test tests both the base class and the test helpers together since
  // they need to be compatible
  helpers::register_derived_with_charm();
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
