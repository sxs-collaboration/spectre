// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>
#include <string>

#include "Parallel/CharmPupable.hpp"

/// \ingroup ComputationalDomainGroup
/// \brief %Domain support for applying boundary conditions
namespace domain::BoundaryConditions {
/*!
 * \brief Base class from which all system-specific base classes must inherit.
 */
class BoundaryCondition : public PUP::able {
 public:
  BoundaryCondition() = default;
  BoundaryCondition(BoundaryCondition&&) = default;
  BoundaryCondition& operator=(BoundaryCondition&&) = default;
  BoundaryCondition(const BoundaryCondition&) = default;
  BoundaryCondition& operator=(const BoundaryCondition&) = default;
  ~BoundaryCondition() override = default;
  explicit BoundaryCondition(CkMigrateMessage* const msg) : PUP::able(msg) {}
  WRAPPED_PUPable_abstract(BoundaryCondition);  // NOLINT

  virtual auto get_clone() const -> std::unique_ptr<BoundaryCondition> = 0;
};
}  // namespace domain::BoundaryConditions
