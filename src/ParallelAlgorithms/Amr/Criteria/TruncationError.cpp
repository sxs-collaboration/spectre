// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Criteria/TruncationError.hpp"

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "Domain/Amr/Flag.hpp"
#include "NumericalAlgorithms/LinearOperators/PowerMonitors.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace amr::Criteria::TruncationError_detail {

template <size_t Dim>
void max_over_components(
    const gsl::not_null<std::array<Flag, Dim>*> result,
    const gsl::not_null<std::array<DataVector, Dim>*> power_monitors_buffer,
    const DataVector& tensor_component, const Mesh<Dim>& mesh,
    const std::optional<double> target_abs_truncation_error,
    const std::optional<double> target_rel_truncation_error) {
  // We take the highest-priority refinement flag in each dimension, so if any
  // tensor component has a truncation error above the target, the element will
  // increase p refinement in that dimension. And only if all tensor components
  // still satisfy the target with the highest mode removed will the element
  // decrease p refinement in that dimension.
  PowerMonitors::power_monitors(power_monitors_buffer, tensor_component, mesh);
  for (size_t d = 0; d < Dim; ++d) {
    // Skip this dimension if we have already decided to refine it
    if (gsl::at(*result, d) == Flag::IncreaseResolution) {
      continue;
    }
    const auto& modes = gsl::at(*power_monitors_buffer, d);
    // Increase p refinement if the truncation error exceeds the target
    const double rel_truncation_error =
        pow(10, -PowerMonitors::relative_truncation_error(modes, modes.size()));
    const double umax = max(abs(tensor_component));
    const double abs_truncation_error = umax * rel_truncation_error;
    if ((target_rel_truncation_error.has_value() and
         rel_truncation_error > target_rel_truncation_error.value()) or
        (target_abs_truncation_error.has_value() and
         abs_truncation_error > target_abs_truncation_error.value())) {
      gsl::at(*result, d) = Flag::IncreaseResolution;
      continue;
    }
    // Dont' check if we want to (allow) decreasing p refinement if another
    // tensor has already decided that decreasing p refinement is bad.
    if (gsl::at(*result, d) == Flag::DoNothing) {
      continue;
    }
    // The `PowerMonitors::relative_truncation_error` function requires at
    // least two modes (and we subtract one below)
    if (modes.size() >= 3) {
      // Decrease p refinement if the truncation error with the highest mode
      // removed still satisfies the target. Otherwise, request to stay at
      // this resolution (or increase resolution if another tensor component
      // requested that).
      const double rel_truncation_error_coarsened = pow(
          10,
          -PowerMonitors::relative_truncation_error(modes, modes.size() - 1));
      const double abs_truncation_error_coarsened =
          umax * rel_truncation_error_coarsened;
      if ((target_rel_truncation_error.has_value() and
           rel_truncation_error_coarsened <=
               target_rel_truncation_error.value()) and
          (target_abs_truncation_error.has_value() and
           abs_truncation_error_coarsened <= target_abs_truncation_error)) {
        gsl::at(*result, d) = Flag::DecreaseResolution;
      } else {
        gsl::at(*result, d) = Flag::DoNothing;
      }
    } else {
      gsl::at(*result, d) = Flag::DoNothing;
    }
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                         \
  template void max_over_components(                                   \
      gsl::not_null<std::array<Flag, DIM(data)>*> result,              \
      const gsl::not_null<std::array<DataVector, DIM(data)>*>          \
          power_monitors_buffer,                                       \
      const DataVector& tensor_component, const Mesh<DIM(data)>& mesh, \
      std::optional<double> target_abs_truncation_error,               \
      std::optional<double> target_rel_truncation_error);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM

}  // namespace amr::Criteria::TruncationError_detail
