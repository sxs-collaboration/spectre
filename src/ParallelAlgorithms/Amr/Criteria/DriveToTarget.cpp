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
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace amr::Criteria {
template <size_t Dim, Type CriteriaType>
DriveToTarget<Dim, CriteriaType>::DriveToTarget(
    const std::array<size_t, Dim>& target,
    const std::array<Flag, Dim>& flags_at_target)
    : target_(target), flags_at_target_(flags_at_target) {
  if constexpr (CriteriaType == Type::h) {
    if (alg::any_of(flags_at_target_, [](Flag flag) {
          return flag == Flag::IncreaseResolution or
                 flag == Flag::DecreaseResolution;
        })) {
      ERROR("Cannot use p-refinement flag in OscillationAtTarget "
            << flags_at_target);
    }
  } else {
    if (alg::any_of(flags_at_target_, [](Flag flag) {
          return flag == Flag::Join or flag == Flag::Split;
        })) {
      ERROR("Cannot use h-refinement flag in OscillationAtTarget "
            << flags_at_target);
    }
  }
}

template <size_t Dim, Type CriteriaType>
DriveToTarget<Dim, CriteriaType>::DriveToTarget(CkMigrateMessage* msg)
    : Criterion(msg) {}

// NOLINTNEXTLINE(google-runtime-references)
template <size_t Dim, Type CriteriaType>
void DriveToTarget<Dim, CriteriaType>::pup(PUP::er& p) {
  Criterion::pup(p);
  p | target_;
  p | flags_at_target_;
}

template <size_t Dim, Type CriteriaType>
std::array<Flag, Dim> DriveToTarget<Dim, CriteriaType>::impl(
    const Mesh<Dim>& current_mesh, const ElementId<Dim>& element_id) const {
  auto result = make_array<Dim>(Flag::DoNothing);
  [[maybe_unused]] const std::array<size_t, Dim> levels =
      element_id.refinement_levels();
  bool is_at_target = true;
  for (size_t d = 0; d < Dim; ++d) {
    if constexpr (CriteriaType == Type::h) {
      if (gsl::at(levels, d) < gsl::at(target_, d)) {
        gsl::at(result, d) = Flag::Split;
        is_at_target = false;
      } else if (gsl::at(levels, d) > gsl::at(target_, d)) {
        gsl::at(result, d) = Flag::Join;
        is_at_target = false;
      }
    } else {
      if (current_mesh.extents(d) < gsl::at(target_, d)) {
        gsl::at(result, d) = Flag::IncreaseResolution;
        is_at_target = false;
      } else if (current_mesh.extents(d) > gsl::at(target_, d)) {
        gsl::at(result, d) = Flag::DecreaseResolution;
        is_at_target = false;
      }
    }
  }
  if (is_at_target) {
    return flags_at_target_;
  }

  return result;
}

#ifndef __CUDA_ARCH__
template <size_t Dim, Type CriteriaType>
PUP::able::PUP_ID DriveToTarget<Dim, CriteriaType>::my_PUP_ID = 0;  // NOLINT
#endif  // __CUDA_ARCH__

template class DriveToTarget<1, Type::h>;
template class DriveToTarget<2, Type::h>;
template class DriveToTarget<3, Type::h>;
template class DriveToTarget<1, Type::p>;
template class DriveToTarget<2, Type::p>;
template class DriveToTarget<3, Type::p>;
}  // namespace amr::Criteria
