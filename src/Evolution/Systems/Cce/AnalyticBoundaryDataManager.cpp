// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/AnalyticBoundaryDataManager.hpp"

#include <cstddef>
#include <utility>

#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"

namespace Cce {
AnalyticBoundaryDataManager::AnalyticBoundaryDataManager(
    const size_t l_max, const double extraction_radius,
    std::unique_ptr<Solutions::WorldtubeData> generator) noexcept
    : l_max_{l_max},
      generator_{std::move(generator)},
      extraction_radius_{extraction_radius} {}

bool AnalyticBoundaryDataManager::populate_hypersurface_boundary_data(
    const gsl::not_null<Variables<
        Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
        boundary_data_variables,
    const double time) const noexcept {
  const auto boundary_tuple = generator_->variables(
      l_max_, time,
      tmpl::list<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>,
                 GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>,
                 GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>{});
  const auto& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>(
          boundary_tuple);
  const auto& pi =
      get<GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>>(boundary_tuple);
  const auto& phi =
      get<GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>(boundary_tuple);
  create_bondi_boundary_data(boundary_data_variables, phi, pi, spacetime_metric,
                             extraction_radius_, l_max_);
  return true;
}

void AnalyticBoundaryDataManager::pup(PUP::er& p) noexcept {
  p | l_max_;
  p | extraction_radius_;
  p | generator_;
}
}  // namespace Cce
