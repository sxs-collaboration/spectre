// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/AmrCriteria/SpecPRefinement.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <pup.h>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/LinearOperators/PowerMonitors.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"

namespace gh::AmrCriteria {
namespace {
// Adds the squared directional power monitors of every component of the tensor
// to the sum of squares.
template <typename TensorType>
void add_squared_power_monitors(
    const gsl::not_null<std::array<DataVector, 3>*> sum_of_squares,
    const gsl::not_null<std::array<DataVector, 3>*> buffer,
    const TensorType& tensor, const Mesh<3>& mesh) {
  for (const auto& component : tensor) {
    PowerMonitors::directional_power_monitors(buffer, component, mesh);
    for (size_t d = 0; d < 3; ++d) {
      gsl::at(*sum_of_squares, d) += square(gsl::at(*buffer, d));
    }
  }
}

// The number of highest modes in each direction that the filter removes,
// measured as SpEC's NumberOfFilteredModes does: the filter is applied to a
// function whose modes in one direction are all one, and the amounts that it
// removes from the modes are added up and rounded.
std::array<size_t, 3> number_of_filtered_modes(
    const Mesh<3>& mesh,
    const Filters::Filter<3, System<3>::variables_tag::tags_list>& filter) {
  auto result = make_array<3>(0_st);
  // A filter that is applied on some steps is applied on step zero.
  if (not(filter.apply_volume_filter_on_substep() or
          filter.apply_volume_filter_on_this_step(0))) {
    return result;
  }
  if (filter.need_jacobians()) {
    ERROR(
        "SpecPRefinement cannot yet determine the number of modes that the "
        "filter '"
        << filter.name() << "' removes, because the filter needs Jacobians.");
  }
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const DataVector ones(number_of_grid_points, 1.0);
  const Matrix identity{};
  // Only one component is needed, the others stay zero.
  Variables<System<3>::variables_tag::tags_list> probe{number_of_grid_points,
                                                       0.0};
  auto& probe_component =
      get<0, 0>(get<gr::Tags::SpacetimeMetric<DataVector, 3>>(probe));
  std::array<DataVector, 3> power{};
  for (size_t d = 0; d < 3; ++d) {
    auto modal_to_nodal = make_array<3>(std::cref(identity));
    gsl::at(modal_to_nodal, d) =
        std::cref(Spectral::modal_to_nodal_matrix(mesh.slice_through(d)));
    probe_component = apply_matrices(modal_to_nodal, ones, mesh.extents());
    filter.apply_in_volume(make_not_null(&probe), mesh, std::nullopt,
                           std::nullopt);
    PowerMonitors::directional_power_monitors(make_not_null(&power),
                                              probe_component, mesh);
    double removed = 0.0;
    for (const double remaining : gsl::at(power, d)) {
      removed += 1.0 - remaining;
    }
    gsl::at(result, d) = static_cast<size_t>(std::round(removed));
  }
  return result;
}
}  // namespace

SpecPRefinement::SpecPRefinement(const double relative_truncation_target)
    : relative_truncation_target_(relative_truncation_target) {}

SpecPRefinement::SpecPRefinement(CkMigrateMessage* msg) : amr::Criterion(msg) {}

std::array<amr::Flag, 3> SpecPRefinement::impl(
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const tnsr::aa<DataVector, 3>& pi, const tnsr::iaa<DataVector, 3>& phi,
    const Mesh<3>& mesh,
    const Filters::Filter<3, System<3>::variables_tag::tags_list>& filter)
    const {
  // The two groups that SpEC monitors: the metric, and Pi and Phi together.
  // The normalization by the number of components is omitted because it
  // cancels in the relative truncation error.
  std::array<std::array<DataVector, 3>, 2> group_powers{};
  for (auto& group_power : group_powers) {
    for (size_t d = 0; d < 3; ++d) {
      gsl::at(group_power, d) = DataVector(mesh.extents(d), 0.0);
    }
  }
  auto& [metric_power, derivative_power] = group_powers;
  std::array<DataVector, 3> buffer{};
  add_squared_power_monitors(make_not_null(&metric_power),
                             make_not_null(&buffer), spacetime_metric, mesh);
  add_squared_power_monitors(make_not_null(&derivative_power),
                             make_not_null(&buffer), pi, mesh);
  add_squared_power_monitors(make_not_null(&derivative_power),
                             make_not_null(&buffer), phi, mesh);

  const auto filtered_modes = number_of_filtered_modes(mesh, filter);
  // SpEC only removes a mode if the truncation error predicted for one less
  // mode is below the target by 0.1 in log10, i.e. by a factor of 10^0.1.
  constexpr double coarsening_margin = 1.2589254117941673;
  auto result = make_array<3>(amr::Flag::DoNothing);
  for (size_t d = 0; d < 3; ++d) {
    if (mesh.extents(d) < gsl::at(filtered_modes, d) + 2) {
      ERROR(
          "SpecPRefinement needs at least two unfiltered modes in every "
          "direction, but direction "
          << d << " of the mesh " << mesh << " has " << mesh.extents(d)
          << " modes of which the filter '" << filter.name() << "' removes "
          << gsl::at(filtered_modes, d) << ".");
    }
    const size_t number_of_unfiltered_modes =
        mesh.extents(d) - gsl::at(filtered_modes, d);
    bool any_group_is_nonzero = false;
    double error = 0.0;
    double error_with_one_less_mode = 0.0;
    for (auto& group_power : group_powers) {
      DataVector& power = gsl::at(group_power, d);
      power = sqrt(power);
      if (max(power) == 0.0) {
        continue;
      }
      any_group_is_nonzero = true;
      const double group_error = PowerMonitors::relative_truncation_error(
          power, number_of_unfiltered_modes);
      error = std::max(error, group_error);
      // With only two unfiltered modes there is no truncation error for one
      // less mode, so the direction is never coarsened.
      error_with_one_less_mode =
          std::max(error_with_one_less_mode,
                   number_of_unfiltered_modes > 2
                       ? std::max(group_error,
                                  PowerMonitors::relative_truncation_error(
                                      power, number_of_unfiltered_modes - 1))
                       : std::numeric_limits<double>::infinity());
    }
    if (not any_group_is_nonzero) {
      continue;
    }
    if (error > relative_truncation_target_) {
      gsl::at(result, d) = amr::Flag::IncreaseResolution;
    } else if (coarsening_margin * error_with_one_less_mode <=
               relative_truncation_target_) {
      gsl::at(result, d) = amr::Flag::DecreaseResolution;
    }
  }
  return result;
}

void SpecPRefinement::pup(PUP::er& p) {
  amr::Criterion::pup(p);
  p | relative_truncation_target_;
}

PUP::able::PUP_ID SpecPRefinement::my_PUP_ID = 0;  // NOLINT
}  // namespace gh::AmrCriteria
