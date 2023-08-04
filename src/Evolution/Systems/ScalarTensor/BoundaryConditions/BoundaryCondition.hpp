// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"

namespace ScalarTensor {
/// \brief Boundary conditions for the combined Generalized Harmonic and
/// CurvedScalarWave systems
namespace BoundaryConditions {

/// \brief The base class for Generalized Harmonic and scalar field combined
/// boundary conditions; all boundary conditions for this system must inherit
/// from this base class.
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
} // namespace BoundaryCondition
} // namespace ScalarTensor
