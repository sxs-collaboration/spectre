// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Utilities/TMPL.hpp"

/// \brief Boundary conditions for the M1Grey radiation transport system
namespace RadiationTransport::M1Grey::BoundaryConditions {

/// \cond
template <typename NeutrinoSpeciesList>
class DirichletAnalytic;
/// \endcond

/// \brief The base class off of which all boundary conditions must inherit
template <typename NeutrinoSpeciesList>
class BoundaryCondition : public domain::BoundaryConditions::BoundaryCondition {
 public:
  using creatable_classes =
      tmpl::list<DirichletAnalytic<NeutrinoSpeciesList>,
                 domain::BoundaryConditions::Periodic<
                     BoundaryCondition<NeutrinoSpeciesList>>>;

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
