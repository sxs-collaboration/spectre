// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Burgers::BoundaryConditions {
class DirichletAnalytic;
class Outflow;
}  // namespace Burgers::BoundaryConditions
/// \endcond

/// \brief Boundary conditions for the Burgers system
namespace Burgers::BoundaryConditions {
/// \brief The base class off of which all boundary conditions must inherit
class BoundaryCondition : public domain::BoundaryConditions::BoundaryCondition {
 public:
  using creatable_classes = tmpl::list<DirichletAnalytic, Outflow>;

  BoundaryCondition() = default;
  BoundaryCondition(BoundaryCondition&&) noexcept = default;
  BoundaryCondition& operator=(BoundaryCondition&&) noexcept = default;
  BoundaryCondition(const BoundaryCondition&) = default;
  BoundaryCondition& operator=(const BoundaryCondition&) = default;
  ~BoundaryCondition() override = default;

  explicit BoundaryCondition(CkMigrateMessage* msg) noexcept;

  void pup(PUP::er& p) override;
};
}  // namespace Burgers::BoundaryConditions
