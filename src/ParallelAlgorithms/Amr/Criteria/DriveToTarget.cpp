// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Criteria/DriveToTarget.hpp"

#include <array>
#include <cstddef>
#include <pup.h>
#include <pup_stl.h>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace amr::Criteria {
template <size_t Dim>
DriveToTarget<Dim>::DriveToTarget(
    const std::array<size_t, Dim>& target_number_of_grid_points,
    const std::array<size_t, Dim>& target_refinement_levels,
    const std::array<Flag, Dim>& flags_at_target)
    : target_number_of_grid_points_(target_number_of_grid_points),
      target_refinement_levels_(target_refinement_levels),
      flags_at_target_(flags_at_target) {}

template <size_t Dim>
DriveToTarget<Dim>::DriveToTarget(CkMigrateMessage* msg) : Criterion(msg) {}

// NOLINTNEXTLINE(google-runtime-references)
template <size_t Dim>
void DriveToTarget<Dim>::pup(PUP::er& p) {
  Criterion::pup(p);
  p | target_number_of_grid_points_;
  p | target_refinement_levels_;
  p | flags_at_target_;
}

template <size_t Dim>
std::array<Flag, Dim> DriveToTarget<Dim>::impl(
    const Mesh<Dim>& current_mesh, const ElementId<Dim>& element_id) const {
  auto result = make_array<Dim>(Flag::DoNothing);
  const std::array<size_t, Dim> levels = element_id.refinement_levels();
  bool is_at_target = true;
  for (size_t d = 0; d < Dim; ++d) {
    if (gsl::at(levels, d) < gsl::at(target_refinement_levels_, d)) {
      gsl::at(result, d) = Flag::Split;
      is_at_target = false;
    } else if (current_mesh.extents(d) <
               gsl::at(target_number_of_grid_points_, d)) {
      gsl::at(result, d) = Flag::IncreaseResolution;
      is_at_target = false;
    } else if (current_mesh.extents(d) >
               gsl::at(target_number_of_grid_points_, d)) {
      gsl::at(result, d) = Flag::DecreaseResolution;
      is_at_target = false;
    } else if (gsl::at(levels, d) > gsl::at(target_refinement_levels_, d)) {
      gsl::at(result, d) = Flag::Join;
      is_at_target = false;
    }
  }
  if (is_at_target) {
    return flags_at_target_;
  }

  return result;
}

template <size_t Dim>
PUP::able::PUP_ID DriveToTarget<Dim>::my_PUP_ID = 0;  // NOLINT

template class DriveToTarget<1>;
template class DriveToTarget<2>;
template class DriveToTarget<3>;
}  // namespace amr::Criteria
