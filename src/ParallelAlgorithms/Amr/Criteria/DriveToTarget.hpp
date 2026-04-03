// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <pup.h>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Tags.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Type.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t>
class ElementId;
template <size_t>
class Mesh;
/// \endcond

namespace amr::Criteria {
/*!
 * \brief Refine the grid towards the target number of grid points and
 * refinement levels in each dimension and then oscillate about the target.
 *
 * \details If the grid is at neither target in a given dimension, the
 * flag chosen will be in the priority order Split, IncreaseResolution,
 * DecreaseResolution, Join.
 *
 * \note To remain at the target, set the OscillationAtTarget Flags to
 * DoNothing.
 *
 * \note This criterion is primarily for testing the mechanics of refinement.
 */
template <size_t Dim, Type CriteriaType>
class DriveToTarget
    : public SPECTRE_CHARM_DERIVED(SINGLE_ARG(DriveToTarget<Dim, CriteriaType>),
                                   Criterion) {
 public:
  /// The target (number of grid points or refinement level) in each dimension
  struct Target {
    using type = std::array<size_t, Dim>;
    static constexpr Options::String help = {
        CriteriaType == Type::p
            ? "The target number of grid points in each dimension."
            : "The target refinement level in each dimension."};
  };

  /// The AMR flags chosen when the target in each dimension is reached
  struct OscillationAtTarget {
    using type = std::array<Flag, Dim>;
    static constexpr Options::String help = {
        "The flags returned when at the target."};
  };

  using options = tmpl::list<Target, OscillationAtTarget>;

  static constexpr Options::String help = {
      "Refine the grid towards the Target, and then oscillate about them by "
      "applying OscillationAtTarget."};

  DriveToTarget() = default;

  DriveToTarget(const std::array<size_t, Dim>& target,
                const std::array<Flag, Dim>& flags_at_target);

  /// \cond
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(DriveToTarget);  // NOLINT
  /// \endcond

  static std::string name() {
    return CriteriaType == Type::p ? "DriveToTargetNumberOfGridPoints"
                                   : "DriveToTargetRefinementLevels";
  }

  Type type() override { return CriteriaType; }

  std::string observation_name() override { return "DriveToTarget"; }

  using compute_tags_for_observation_box = tmpl::list<>;

  using argument_tags = tmpl::list<::domain::Tags::Mesh<Dim>>;

  template <typename Metavariables>
  auto operator()(const Mesh<Dim>& current_mesh,
                  Parallel::GlobalCache<Metavariables>& /*cache*/,
                  const ElementId<Dim>& element_id) const;

  void pup(PUP::er& p) override;

 private:
  std::array<Flag, Dim> impl(const Mesh<Dim>& current_mesh,
                             const ElementId<Dim>& element_id) const;

  std::array<size_t, Dim> target_{};
  std::array<Flag, Dim> flags_at_target_{};
};

template <size_t Dim, Type CriteriaType>
template <typename Metavariables>
auto DriveToTarget<Dim, CriteriaType>::operator()(
    const Mesh<Dim>& current_mesh,
    Parallel::GlobalCache<Metavariables>& /*cache*/,
    const ElementId<Dim>& element_id) const {
  return impl(current_mesh, element_id);
}
}  // namespace amr::Criteria
