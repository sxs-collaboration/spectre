// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/AnalyticBoundaryDataManager.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce {
AnalyticBoundaryDataManager::AnalyticBoundaryDataManager(
    const size_t l_max, const double extraction_radius,
    std::unique_ptr<Solutions::WorldtubeData> generator,
    const double du_dr_j_stencil_step)
    : l_max_{l_max},
      generator_{std::move(generator)},
      extraction_radius_{extraction_radius},
      du_dr_j_stencil_step_{du_dr_j_stencil_step} {}

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
  return true;
}

void AnalyticBoundaryDataManager::populate_du_dr_j(
    const gsl::not_null<Variables<
        Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
        boundary_data_variables,
    const double time) const {
  if (du_dr_j_interpolator_ == nullptr) {
    return;
  }
  const size_t stencil_size =
      2 * du_dr_j_interpolator_->required_number_of_points_before_and_after();
  // An even number of points centered on `time` has half-integer offsets, so
  // `time` itself is not among the samples.
  const double first_offset = -0.5 * static_cast<double>(stencil_size - 1);
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max_);

  DataVector stencil_times{stencil_size};
  ComplexDataVector dr_j_over_time{stencil_size * number_of_angular_points};
  Variables<Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>
      stencil_buffer{number_of_angular_points};
  for (size_t i = 0; i < stencil_size; ++i) {
    stencil_times[i] =
        time + (first_offset + static_cast<double>(i)) * du_dr_j_stencil_step_;
    populate_hypersurface_boundary_data(make_not_null(&stencil_buffer),
                                        stencil_times[i]);
    const auto& stencil_dr_j =
        get(get<Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>>(stencil_buffer))
            .data();
    for (size_t k = 0; k < number_of_angular_points; ++k) {
      dr_j_over_time[(i * number_of_angular_points) + k] = stencil_dr_j[k];
    }
  }
  detail::differentiate_dr_j_in_time(
      make_not_null(
          &get(get<Tags::BoundaryValue<Tags::Du<Tags::Dr<Tags::BondiJ>>>>(
                   *boundary_data_variables))
               .data()),
      stencil_times, dr_j_over_time, *du_dr_j_interpolator_, time);
}

void AnalyticBoundaryDataManager::set_du_dr_j_interpolator(
    std::unique_ptr<intrp::SpanInterpolator> interpolator) {
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
