// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <pup.h>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Tags.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
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
template <size_t Dim>
class DriveToTarget : public Criterion {
 public:
  /// The target number of grid point in each dimension
  struct TargetNumberOfGridPoints {
    using type = std::array<size_t, Dim>;
    static constexpr Options::String help = {
        "The target number of grid points in each dimension."};
  };

  /// The target refinement level in each dimension
  struct TargetRefinementLevels {
    using type = std::array<size_t, Dim>;
    static constexpr Options::String help = {
        "The target refinement level in each dimension."};
  };

  /// The AMR flags chosen when the target number of grid points and refinement
  /// levels are reached
  struct OscillationAtTarget {
    using type = std::array<Flag, Dim>;
    static constexpr Options::String help = {
        "The flags returned when at the target."};
  };

  using options = tmpl::list<TargetNumberOfGridPoints, TargetRefinementLevels,
                             OscillationAtTarget>;

  static constexpr Options::String help = {
      "Refine the grid towards the TargetNumberOfGridPoints and "
      "TargetRefinementLevels, and then oscillate about them by applying "
      "OscillationAtTarget."};

  DriveToTarget() = default;

  DriveToTarget(const std::array<size_t, Dim>& target_number_of_grid_points,
                const std::array<size_t, Dim>& target_refinement_levels,
                const std::array<Flag, Dim>& flags_at_target);

  /// \cond
  explicit DriveToTarget(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(DriveToTarget);  // NOLINT
  /// \endcond

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

  std::array<size_t, Dim> target_number_of_grid_points_{};
  std::array<size_t, Dim> target_refinement_levels_{};
  std::array<Flag, Dim> flags_at_target_{};
};

template <size_t Dim>
template <typename Metavariables>
auto DriveToTarget<Dim>::operator()(
    const Mesh<Dim>& current_mesh,
    Parallel::GlobalCache<Metavariables>& /*cache*/,
    const ElementId<Dim>& element_id) const {
  return impl(current_mesh, element_id);
}
}  // namespace amr::Criteria
