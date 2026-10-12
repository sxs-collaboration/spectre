// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/AnalyticBoundaryDataManager.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/RobinsonTrautman.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce {
AnalyticBoundaryDataManager::AnalyticBoundaryDataManager(
    const size_t l_max, const double extraction_radius,
    std::unique_ptr<Solutions::WorldtubeData> generator,
    const double du_dr_j_stencil_step)
    : l_max_{l_max},
      generator_{std::move(generator)},
      extraction_radius_{extraction_radius},
      du_dr_j_stencil_step_{du_dr_j_stencil_step} {
  if (not std::isfinite(du_dr_j_stencil_step_) or
      du_dr_j_stencil_step_ <= 0.0) {
    ERROR("DuDrJStencilStep must be finite and positive, but is "
          << du_dr_j_stencil_step_);
  }
}

bool AnalyticBoundaryDataManager::populate_hypersurface_boundary_data(
    const gsl::not_null<Variables<
        Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
        boundary_data_variables,
    const double time) const {
  const auto boundary_tuple = generator_->variables(
      l_max_, time,
      tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
                 gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>>{});
  const auto& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<DataVector, 3>>(boundary_tuple);
  const auto& pi = get<gh::Tags::Pi<DataVector, 3>>(boundary_tuple);
  const auto& phi = get<gh::Tags::Phi<DataVector, 3>>(boundary_tuple);
  create_bondi_boundary_data(boundary_data_variables, phi, pi, spacetime_metric,
                             extraction_radius_, l_max_);

  // `create_bondi_boundary_data` fills every boundary tag except
  // `Du<Dr<BondiJ>>`, so sample `Dr<BondiJ>` on a uniform time stencil
  // centered on `time` and differentiate it.
  if (du_dr_j_interpolator_ != nullptr) {
    const size_t stencil_size =
        2 * du_dr_j_interpolator_->required_number_of_points_before_and_after();
    const double max_offset = 0.5 * static_cast<double>(stencil_size - 1);
    const double max_time = std::numeric_limits<double>::max();
    if (not std::isfinite(time) or
        du_dr_j_stencil_step_ > max_time / std::max(1.0, max_offset) or
        abs(time) > max_time - max_offset * du_dr_j_stencil_step_) {
      ERROR("DuDrJStencilStep " << du_dr_j_stencil_step_
                                << " does not give distinct, finite stencil "
                                   "times around "
                                << time);
    }
    const size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max_);

    DataVector stencil_times{stencil_size};
    ComplexDataVector dr_j_over_time{stencil_size * number_of_angular_points};
    Variables<Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>
        stencil_buffer{number_of_angular_points};
    for (size_t i = 0; i < stencil_size; ++i) {
      // An even number of points centered on `time` has half-integer offsets,
      // so `time` itself is not among the samples.
      const double offset = static_cast<double>(i) - max_offset;
      stencil_times[i] = time + offset * du_dr_j_stencil_step_;
      if (not std::isfinite(stencil_times[i]) or
          (i > 0 and stencil_times[i] <= stencil_times[i - 1])) {
        ERROR("DuDrJStencilStep " << du_dr_j_stencil_step_
                                  << " does not give distinct, finite stencil "
                                     "times around "
                                  << time);
      }
      const auto stencil_tuple = generator_->variables(
          l_max_, stencil_times[i],
          tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
                     gh::Tags::Pi<DataVector, 3>,
                     gh::Tags::Phi<DataVector, 3>>{});
      create_bondi_boundary_data(
          make_not_null(&stencil_buffer),
          get<gh::Tags::Phi<DataVector, 3>>(stencil_tuple),
          get<gh::Tags::Pi<DataVector, 3>>(stencil_tuple),
          get<gr::Tags::SpacetimeMetric<DataVector, 3>>(stencil_tuple),
          extraction_radius_, l_max_);
      const auto& stencil_dr_j =
          get(get<Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>>(stencil_buffer))
              .data();
      for (size_t k = 0; k < number_of_angular_points; ++k) {
        dr_j_over_time[(i * number_of_angular_points) + k] = stencil_dr_j[k];
      }
    }

    auto& du_dr_j =
        get(get<Tags::BoundaryValue<Tags::Du<Tags::Dr<Tags::BondiJ>>>>(
                *boundary_data_variables))
            .data();
    ComplexDataVector dr_j_at_point{stencil_size};
    for (size_t k = 0; k < number_of_angular_points; ++k) {
      for (size_t i = 0; i < stencil_size; ++i) {
        dr_j_at_point[i] = dr_j_over_time[(i * number_of_angular_points) + k];
      }
      du_dr_j[k] = du_dr_j_interpolator_->derivative(
          gsl::span<const double>(stencil_times.data(), stencil_times.size()),
          gsl::span<const std::complex<double>>(dr_j_at_point.data(),
                                                dr_j_at_point.size()),
          time);
    }
  }
  return true;
}

void AnalyticBoundaryDataManager::set_du_dr_j_interpolator(
    std::unique_ptr<intrp::SpanInterpolator> interpolator) {
  if (interpolator != nullptr and
      dynamic_cast<const Solutions::RobinsonTrautman*>(generator_.get()) !=
          nullptr) {
    ERROR(
        "RobinsonTrautman does not support the centered time stencil for "
        "Du<Dr<BondiJ>>: past-time samples restart its internal evolution "
        "and can precede its initial time. Use an initial-data generator "
        "that does not require Du<Dr<BondiJ>>.");
  }
  du_dr_j_interpolator_ = std::move(interpolator);
}

void AnalyticBoundaryDataManager::pup(PUP::er& p) {
  p | l_max_;
  p | extraction_radius_;
  p | generator_;
  p | du_dr_j_interpolator_;
  p | du_dr_j_stencil_step_;
}
}  // namespace Cce
