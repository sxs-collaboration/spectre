// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace GeneralizedHarmonic::BoundaryConditions {
template <size_t Dim>
class ConstraintPreservingBjorhus;
template <size_t Dim>
class DirichletAnalytic;
template <size_t Dim>
class FreezingBjorhus;
template <size_t Dim>
class Outflow;
}  // namespace GeneralizedHarmonic::BoundaryConditions
/// \endcond

/// \brief Boundary conditions for the generalized harmonic system
namespace GeneralizedHarmonic::BoundaryConditions {
/// \brief The base class off of which all boundary conditions must inherit
template <size_t Dim>
class BoundaryCondition : public domain::BoundaryConditions::BoundaryCondition {
 public:
  using creatable_classes =
      tmpl::list<ConstraintPreservingBjorhus<Dim>, DirichletAnalytic<Dim>,
                 FreezingBjorhus<Dim>, Outflow<Dim>,
                 domain::BoundaryConditions::Periodic<BoundaryCondition<Dim>>>;

  BoundaryCondition() = default;
  BoundaryCondition(BoundaryCondition&&) noexcept = default;
  BoundaryCondition& operator=(BoundaryCondition&&) noexcept = default;
  BoundaryCondition(const BoundaryCondition&) = default;
  BoundaryCondition& operator=(const BoundaryCondition&) = default;
  ~BoundaryCondition() override = default;
  explicit BoundaryCondition(CkMigrateMessage* msg) noexcept;

  void pup(PUP::er& p) override;
};
}  // namespace GeneralizedHarmonic::BoundaryConditions
