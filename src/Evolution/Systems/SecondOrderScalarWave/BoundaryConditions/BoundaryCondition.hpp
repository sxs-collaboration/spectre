// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// \brief Boundary conditions for the second-order scalar wave system
namespace SecondOrderScalarWave::BoundaryConditions {
/// \brief The base class off of which all boundary conditions must inherit
template <size_t Dim>
class BoundaryCondition : public domain::BoundaryConditions::BoundaryCondition {
 public:
  BoundaryCondition() = default;
  BoundaryCondition(BoundaryCondition&&) = default;
  BoundaryCondition& operator=(BoundaryCondition&&) = default;
  BoundaryCondition(const BoundaryCondition&) = default;
  BoundaryCondition& operator=(const BoundaryCondition&) = default;
  ~BoundaryCondition() override = default;
  explicit BoundaryCondition(CkMigrateMessage* msg);

  void pup(PUP::er& p) override;
};
}  // namespace SecondOrderScalarWave::BoundaryConditions
