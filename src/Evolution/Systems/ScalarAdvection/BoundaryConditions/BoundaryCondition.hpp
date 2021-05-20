// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarAdvection {
/// \brief Boundary conditions for the ScalarAdvection system
namespace BoundaryConditions {
/// \brief The base class off of which all boundary conditions must inherit
template <size_t Dim>
class BoundaryCondition : public domain::BoundaryConditions::BoundaryCondition {
 public:
  using creatable_classes =
      tmpl::list<domain::BoundaryConditions::Periodic<BoundaryCondition<Dim>>>;

  BoundaryCondition() = default;
  BoundaryCondition(BoundaryCondition&&) noexcept = default;
  BoundaryCondition& operator=(BoundaryCondition&&) noexcept = default;
  BoundaryCondition(const BoundaryCondition&) = default;
  BoundaryCondition& operator=(const BoundaryCondition&) = default;
  ~BoundaryCondition() override = default;

  explicit BoundaryCondition(CkMigrateMessage* msg) noexcept;

  void pup(PUP::er& p) override;
};
}  // namespace BoundaryConditions
}  // namespace ScalarAdvection
