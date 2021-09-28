// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::BoundaryConditions {
/// Mark a boundary condition as being periodic.
///
/// Periodic boundary conditions shouldn't require any implementation outside of
/// a check in the domain creator using the `is_periodic()` function to
/// determine what boundaries are periodic. Across each matching pair of
/// periodic boundary conditions, the domain creator should specify that the DG
/// elements are neighbors of each other.
class MarkAsPeriodic {
 public:
  MarkAsPeriodic() = default;
  MarkAsPeriodic(MarkAsPeriodic&&) = default;
  MarkAsPeriodic& operator=(MarkAsPeriodic&&) = default;
  MarkAsPeriodic(const MarkAsPeriodic&) = default;
  MarkAsPeriodic& operator=(const MarkAsPeriodic&) = default;
  virtual ~MarkAsPeriodic() = 0;
};

/*!
 * \brief Periodic boundary conditions.
 *
 * To use with a specific system add:
 *
 * \code
 *  domain::BoundaryConditions::Periodic<your::system::BoundaryConditionBase>
 * \endcode
 *
 * to the list of creatable classes.
 *
 * \note Not all domain creators will allow you to specify periodic boundary
 * conditions since they may not make sense.
 */
template <typename SystemBoundaryConditionBaseClass>
struct Periodic final : public SystemBoundaryConditionBaseClass,
                        public MarkAsPeriodic {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Periodic boundary conditions.\n\nNote: Not all domain creators will "
      "allow you to specify periodic boundary conditions since they may not "
      "make sense."};
  static std::string name() { return "Periodic"; }

  Periodic() = default;
  Periodic(Periodic&&) = default;
  Periodic& operator=(Periodic&&) = default;
  Periodic(const Periodic&) = default;
  Periodic& operator=(const Periodic&) = default;
  ~Periodic() override = default;

  explicit Periodic(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, Periodic);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  void pup(PUP::er& p) override;
};

template <typename SystemBoundaryConditionBaseClass>
Periodic<SystemBoundaryConditionBaseClass>::Periodic(
    CkMigrateMessage* const msg)
    : SystemBoundaryConditionBaseClass(msg) {}

template <typename SystemBoundaryConditionBaseClass>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Periodic<SystemBoundaryConditionBaseClass>::get_clone() const {
  return std::make_unique<Periodic>(*this);
}

template <typename SystemBoundaryConditionBaseClass>
void Periodic<SystemBoundaryConditionBaseClass>::pup(PUP::er& p) {
  BoundaryCondition::pup(p);
}

/// \cond
template <typename SystemBoundaryConditionBaseClass>
// NOLINTNEXTLINE
PUP::able::PUP_ID Periodic<SystemBoundaryConditionBaseClass>::my_PUP_ID = 0;
/// \endcond

/// Check if a boundary condition inherits from `MarkAsPeriodic`, which
/// constitutes as it being marked as a periodic boundary condition.
bool is_periodic(const std::unique_ptr<BoundaryCondition>& boundary_condition);
}  // namespace domain::BoundaryConditions
