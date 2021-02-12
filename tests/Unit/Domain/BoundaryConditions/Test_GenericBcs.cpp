// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <pup.h>
#include <string>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Parallel/CharmPupable.hpp"

namespace {
namespace helpers = TestHelpers::domain::BoundaryConditions;

SPECTRE_TEST_CASE("Unit.Domain.BoundaryConditions.GenericBcs",
                  "[Unit][Domain]") {
  helpers::register_derived_with_charm();

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> periodic =
      std::make_unique<helpers::TestPeriodicBoundaryCondition<1>>();
  CHECK(is_periodic(periodic));
  CHECK(is_periodic(serialize_and_deserialize(periodic)));
  CHECK_FALSE(is_none(periodic));
  CHECK_FALSE(is_none(serialize_and_deserialize(periodic)));

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> not_periodic =
      std::make_unique<helpers::TestBoundaryCondition<1>>();
  CHECK_FALSE(is_periodic(not_periodic));
  CHECK_FALSE(is_periodic(serialize_and_deserialize(not_periodic)));
  CHECK_FALSE(is_none(not_periodic));
  CHECK_FALSE(is_none(serialize_and_deserialize(not_periodic)));

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> none =
      std::make_unique<helpers::TestNoneBoundaryCondition<1>>();
  CHECK(is_none(none));
  CHECK(is_none(serialize_and_deserialize(none)));
  CHECK_FALSE(is_periodic(none));
  CHECK_FALSE(is_periodic(serialize_and_deserialize(none)));
}
}  // namespace
