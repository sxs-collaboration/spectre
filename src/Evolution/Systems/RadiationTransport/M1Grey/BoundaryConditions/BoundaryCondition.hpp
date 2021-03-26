// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace RadiationTransport::M1Grey::BoundaryConditions {
template <typename... NeutrinoSpecies>
class DirichletAnalytic;
}  // namespace RadiationTransport::M1Grey::BoundaryConditions
/// \endcond

/// \brief Boundary conditions for the M1Grey radiation transport system
namespace RadiationTransport::M1Grey::BoundaryConditions {
/// \brief The base class off of which all boundary conditions must inherit
template <typename... NeutrinoSpecies>
class BoundaryCondition : public domain::BoundaryConditions::BoundaryCondition {
 public:
  using creatable_classes = tmpl::list<DirichletAnalytic<NeutrinoSpecies...>>;

  BoundaryCondition() = default;
  BoundaryCondition(BoundaryCondition&&) noexcept = default;
  BoundaryCondition& operator=(BoundaryCondition&&) noexcept = default;
  BoundaryCondition(const BoundaryCondition&) = default;
  BoundaryCondition& operator=(const BoundaryCondition&) = default;
  ~BoundaryCondition() override = default;

  explicit BoundaryCondition(CkMigrateMessage* const msg) noexcept
      : domain::BoundaryConditions::BoundaryCondition(msg) {}

  void pup(PUP::er& p) override {
    domain::BoundaryConditions::BoundaryCondition::pup(p);
  }
};
}  // namespace RadiationTransport::M1Grey::BoundaryConditions
