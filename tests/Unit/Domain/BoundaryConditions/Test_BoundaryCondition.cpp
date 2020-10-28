// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <pup.h>
#include <string>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/CharmPupable.hpp"

namespace {
class BoundaryCondition
    : public domain::BoundaryConditions::BoundaryCondition {
 public:
  BoundaryCondition() = default;

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const noexcept override {
    return std::make_unique<BoundaryCondition>(*this);
  }

  explicit BoundaryCondition(CkMigrateMessage* msg) noexcept
      : domain::BoundaryConditions::BoundaryCondition(msg) {}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, BoundaryCondition);
#pragma GCC diagnostic pop

  void pup(PUP::er& p) override {
    domain::BoundaryConditions::BoundaryCondition::pup(p);
  }
};

PUP::able::PUP_ID BoundaryCondition::my_PUP_ID = 0;

SPECTRE_TEST_CASE("Unit.Domain.BoundaryConditions.BoundaryCondition",
                  "[Unit][Domain]") {
  PUPable_reg(BoundaryCondition);
  const std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> bc =
      std::make_unique<BoundaryCondition>();
  const auto bc_deserialized = serialize_and_deserialize(bc);
  CHECK(dynamic_cast<const BoundaryCondition* const>(bc_deserialized.get()) !=
        nullptr);
}
}  // namespace
