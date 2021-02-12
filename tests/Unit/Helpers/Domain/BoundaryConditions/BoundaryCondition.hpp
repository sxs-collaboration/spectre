// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup TestingFrameworkGroup
/// \brief Helpers for boundary conditions
namespace TestHelpers::domain::BoundaryConditions {
/// \cond
template <size_t Dim>
class TestBoundaryCondition;
/// \endcond

/// \brief A system-specific boundary condition base class.
///
/// To be used in conjunction with `SystemWithBoundaryConditions`
template <size_t Dim>
class BoundaryConditionBase
    : public ::domain::BoundaryConditions::BoundaryCondition {
 public:
  using creatable_classes = tmpl::list<
      TestBoundaryCondition<Dim>,
      ::domain::BoundaryConditions::Periodic<BoundaryConditionBase<Dim>>>;

  BoundaryConditionBase() = default;
  BoundaryConditionBase(BoundaryConditionBase&&) noexcept = default;
  BoundaryConditionBase& operator=(BoundaryConditionBase&&) noexcept = default;
  BoundaryConditionBase(const BoundaryConditionBase&) = default;
  BoundaryConditionBase& operator=(const BoundaryConditionBase&) = default;
  ~BoundaryConditionBase() override = default;

  explicit BoundaryConditionBase(CkMigrateMessage* msg) noexcept;

  void pup(PUP::er& p) override;

  static constexpr Options::String help = {"Boundary conditions for tests."};
};

/// \brief Concrete boundary condition
template <size_t Dim>
class TestBoundaryCondition final : public BoundaryConditionBase<Dim> {
 public:
  TestBoundaryCondition() = default;
  explicit TestBoundaryCondition(Direction<Dim> direction, size_t block_id = 0);
  TestBoundaryCondition(const std::string& direction, size_t block_id);
  TestBoundaryCondition(TestBoundaryCondition&&) noexcept = default;
  TestBoundaryCondition& operator=(TestBoundaryCondition&&) noexcept = default;
  TestBoundaryCondition(const TestBoundaryCondition&) = default;
  TestBoundaryCondition& operator=(const TestBoundaryCondition&) = default;
  ~TestBoundaryCondition() override = default;
  explicit TestBoundaryCondition(CkMigrateMessage* const msg) noexcept;

  struct DirectionOptionTag {
    using type = std::string;
    static std::string name() noexcept { return "Direction"; }
    static constexpr Options::String help =
        "The direction the boundary condition operates in.";
  };
  struct BlockIdOptionTag {
    using type = size_t;
    static std::string name() noexcept { return "BlockId"; }
    static constexpr Options::String help =
        "The id of the block the boundary condition operates in.";
  };

  using options = tmpl::list<DirectionOptionTag, BlockIdOptionTag>;

  static constexpr Options::String help = {"Boundary condition for testing."};

  WRAPPED_PUPable_decl_base_template(
      ::domain::BoundaryConditions::BoundaryCondition,
      TestBoundaryCondition<Dim>);

  const Direction<Dim>& direction() const noexcept { return direction_; }
  size_t block_id() const noexcept { return block_id_; }

  auto get_clone() const noexcept -> std::unique_ptr<
      ::domain::BoundaryConditions::BoundaryCondition> override;

  void pup(PUP::er& p) override;

 private:
  Direction<Dim> direction_{};
  size_t block_id_{0};
};

template <size_t Dim>
bool operator==(const TestBoundaryCondition<Dim>& lhs,
                const TestBoundaryCondition<Dim>& rhs) noexcept;

template <size_t Dim>
bool operator!=(const TestBoundaryCondition<Dim>& lhs,
                const TestBoundaryCondition<Dim>& rhs) noexcept;

template <size_t Dim>
using TestPeriodicBoundaryCondition =
    ::domain::BoundaryConditions::Periodic<BoundaryConditionBase<Dim>>;

/// Empty system that has boundary conditions
template <size_t Dim>
struct SystemWithBoundaryConditions {
  using boundary_conditions_base = BoundaryConditionBase<Dim>;
};

/// Empty system that doesn't have boundary conditions
template <size_t Dim>
struct SystemWithoutBoundaryConditions {};

/// Metavariables with a system that has boundary conditions
template <size_t Dim>
struct MetavariablesWithBoundaryConditions {
  using system = SystemWithBoundaryConditions<Dim>;
};

/// Metavariables with a system that doesn't have boundary conditions
template <size_t Dim>
struct MetavariablesWithoutBoundaryConditions {
  using system = SystemWithoutBoundaryConditions<Dim>;
};

void register_derived_with_charm() noexcept;
}  // namespace TestHelpers::domain::BoundaryConditions
