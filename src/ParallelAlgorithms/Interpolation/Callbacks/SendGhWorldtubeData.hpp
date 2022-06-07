// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/Actions/ReceiveGhWorldtubeData.hpp"
#include "Evolution/Systems/Cce/Components/WorldtubeBoundary.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/PostInterpolationCallback.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace intrp {
namespace callbacks {

/// \brief post_interpolation_callback that calls Cce::ReceiveGhWorldTubeData
///
/// Uses:
/// - DataBox:
///   - `::gr::Tags::SpacetimeMetric<3,Frame::Inertial>`
///   - `::GeneralizedHarmonic::Tags::Pi<3,Frame::Inertial>`
///   - `::GeneralizedHarmonic::Tags::Phi<3,Frame::Inertial>`
///
/// Conforms to the intrp::protocols::PostInterpolationCallback protocol
///
/// For requirements on InterpolationTargetTag, see
/// intrp::protocols::InterpolationTargetTag
///
/// \note This callback requires the temporal ID in an InterpolationTargetTag be
/// a TimeStepId.
template <typename CceEvolutionComponent, bool DuringSelfStart>
struct SendGhWorldtubeData
    : tt::ConformsTo<intrp::protocols::PostInterpolationCallback> {
  template <typename DbTags, typename Metavariables, typename TemporalId>
  static void apply(const db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const TemporalId& temporal_id) {
    auto& cce_gh_boundary_component = Parallel::get_parallel_component<
        Cce::GhWorldtubeBoundary<Metavariables>>(cache);
    Parallel::simple_action<typename Cce::Actions::ReceiveGhWorldtubeData<
        CceEvolutionComponent, DuringSelfStart>>(
        cce_gh_boundary_component, temporal_id,
        db::get<::gr::Tags::SpacetimeMetric<3, Frame::Inertial>>(box),
        db::get<::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>(box),
        db::get<::GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>(box));
  }
};
}  // namespace callbacks
}  // namespace intrp
