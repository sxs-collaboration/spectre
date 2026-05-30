// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <pup.h>
#include <string>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/Cartoon.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace {
namespace helpers = TestHelpers::domain::BoundaryConditions;

SPECTRE_TEST_CASE("Unit.Domain.BoundaryConditions.GenericBcs",
                  "[Unit][Domain]") {
  helpers::register_derived_with_charm();

  const std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      periodic = std::make_unique<helpers::TestPeriodicBoundaryCondition<1>>();
  CHECK(is_periodic(periodic));
  CHECK(is_periodic(serialize_and_deserialize(periodic)));
  CHECK_FALSE(is_none(periodic));
  CHECK_FALSE(is_none(serialize_and_deserialize(periodic)));
  CHECK_FALSE(is_cartoon(periodic));
  CHECK_FALSE(is_cartoon(serialize_and_deserialize(periodic)));

  const std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      not_periodic = std::make_unique<helpers::TestBoundaryCondition<1>>();
  CHECK_FALSE(is_periodic(not_periodic));
  CHECK_FALSE(is_periodic(serialize_and_deserialize(not_periodic)));
  CHECK_FALSE(is_none(not_periodic));
  CHECK_FALSE(is_none(serialize_and_deserialize(not_periodic)));
  CHECK_FALSE(is_cartoon(not_periodic));
  CHECK_FALSE(is_cartoon(serialize_and_deserialize(not_periodic)));

  const std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> cartoon =
      std::make_unique<helpers::TestCartoonBoundaryCondition<3>>();
  CHECK(is_cartoon(cartoon));
  CHECK(is_cartoon(serialize_and_deserialize(cartoon)));
  CHECK_FALSE(is_none(cartoon));
  CHECK_FALSE(is_none(serialize_and_deserialize(cartoon)));
  CHECK_FALSE(is_periodic(cartoon));
  CHECK_FALSE(is_periodic(serialize_and_deserialize(cartoon)));

  const std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      not_cartoon = std::make_unique<helpers::TestBoundaryCondition<1>>();
  CHECK_FALSE(is_cartoon(not_cartoon));
  CHECK_FALSE(is_cartoon(serialize_and_deserialize(not_cartoon)));
  CHECK_FALSE(is_none(not_cartoon));
  CHECK_FALSE(is_none(serialize_and_deserialize(not_cartoon)));
  CHECK_FALSE(is_periodic(not_cartoon));
  CHECK_FALSE(is_periodic(serialize_and_deserialize(not_cartoon)));

  const std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> none =
      std::make_unique<helpers::TestNoneBoundaryCondition<1>>();
  CHECK(is_none(none));
  CHECK(is_none(serialize_and_deserialize(none)));
  CHECK_FALSE(is_periodic(none));
  CHECK_FALSE(is_periodic(serialize_and_deserialize(none)));
  CHECK_FALSE(is_cartoon(none));
  CHECK_FALSE(is_cartoon(serialize_and_deserialize(none)));
}
}  // namespace
