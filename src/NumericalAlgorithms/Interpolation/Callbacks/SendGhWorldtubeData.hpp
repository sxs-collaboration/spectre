// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/Actions/ReceiveGhWorldtubeData.hpp"
#include "Evolution/Systems/Cce/Components/WorldtubeBoundary.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace intrp {
namespace callbacks {

/// \brief post_interpolation_callback that calls Cce::ReceiveGhWorldTubeData
///
/// Uses:
/// - Metavariables
///   - `temporal_id`
/// - DataBox:
///   - `::gr::Tags::SpacetimeMetric<3,Frame::Inertial>`
///   - `::GeneralizedHarmonic::Tags::Pi<3,Frame::Inertial>`
///   - `::GeneralizedHarmonic::Tags::Phi<3,Frame::Inertial>`
///
/// This is an InterpolationTargetTag::post_interpolation_callback;
/// see InterpolationTarget for a description of InterpolationTargetTag.
template <typename CceEvolutionComponent>
struct SendGhWorldtubeData {
  using observation_types = tmpl::list<>;
  template <typename DbTags, typename Metavariables>
  static void apply(
      const db::DataBox<DbTags>& box,
      Parallel::GlobalCache<Metavariables>& cache,
      const typename Metavariables::temporal_id::type& temporal_id) noexcept {
    auto& cce_gh_boundary_component = Parallel::get_parallel_component<
        Cce::GhWorldtubeBoundary<Metavariables>>(cache);
    Parallel::simple_action<
        typename Cce::Actions::ReceiveGhWorldtubeData<CceEvolutionComponent>>(
        cce_gh_boundary_component, temporal_id,
        db::get<::gr::Tags::SpacetimeMetric<3, Frame::Inertial>>(box),
        db::get<::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>(box),
        db::get<::GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>(box),
        db::get<::Tags::dt<::gr::Tags::SpacetimeMetric<3, Frame::Inertial>>>(
            box),
        db::get<
            ::Tags::dt<::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>>(
            box),
        db::get<
            ::Tags::dt<::GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>>(
            box));
  }
};
}  // namespace callbacks
}  // namespace intrp
