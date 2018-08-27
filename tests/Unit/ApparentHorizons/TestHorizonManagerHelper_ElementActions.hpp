// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>
#include <utility>

#include "ApparentHorizons/HorizonManagerComponentActions.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Algorithm.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "PointwiseFunctions/AnalyticSolutions/EinsteinSolutions/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/GrTags.hpp"
#include "Time/Time.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace test_ah {
namespace Actions {
namespace DgElementArray {

struct InitializeElement {
  using return_tag_list = tmpl::list<Tags::Mesh<3>, ah::vars_tags_from_element>;

  template <typename... InboxTags, typename Metavariables, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const std::vector<std::array<size_t, 3>>& initial_extents,
                    const Domain<3, Frame::Inertial>& domain) noexcept {
    ElementId<3> element_id{array_index};
    const auto& my_block = domain.blocks()[element_id.block_id()];
    ::Mesh<3> mesh{initial_extents[element_id.block_id()],
                   Spectral::Basis::Legendre,
                   Spectral::Quadrature::GaussLobatto};

    // Coordinates
    ElementMap<3, Frame::Inertial> map{element_id,
                                       my_block.coordinate_map().get_clone()};
    auto inertial_coords = map(logical_coordinates(mesh));

    EinsteinSolutions::KerrSchild solution(1.0, {{0., 0., 0.}}, {{0., 0., 0.}});
    const auto input_vars =
        solution.variables(inertial_coords, 0.0,
                           EinsteinSolutions::KerrSchild::tags<DataVector>{});
    const auto& lapse =
        get<gr::Tags::Lapse<3, Frame::Inertial, DataVector>>(input_vars);
    const auto& dt_lapse =
        get<Tags::dt<gr::Tags::Lapse<3, Frame::Inertial, DataVector>>>(
            input_vars);
    const auto& d_lapse =
        get<EinsteinSolutions::KerrSchild::DerivLapse<DataVector>>(input_vars);
    const auto& shift =
        get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(input_vars);
    const auto& d_shift =
        get<EinsteinSolutions::KerrSchild::DerivShift<DataVector>>(input_vars);
    const auto& dt_shift =
        get<Tags::dt<gr::Tags::Shift<3, Frame::Inertial, DataVector>>>(
            input_vars);
    const auto& g =
        get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(
            input_vars);
    const auto& dt_g =
        get<Tags::dt<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>>(
            input_vars);
    const auto& d_g =
        get<EinsteinSolutions::KerrSchild::DerivSpatialMetric<DataVector>>(
            input_vars);

    db::item_type<ah::vars_tags_from_element> output_vars(
        mesh.number_of_grid_points());
    auto& psi = get<::gr::Tags::SpacetimeMetric<3>>(output_vars);
    auto& pi = get<::GeneralizedHarmonic::Pi<3>>(output_vars);
    auto& phi = get<::GeneralizedHarmonic::Phi<3>>(output_vars);
    psi = gr::spacetime_metric(lapse, shift, g);
    phi = GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift, g, d_g);
    pi =
        GeneralizedHarmonic::pi(lapse, dt_lapse, shift, dt_shift, g, dt_g, phi);

    db::compute_databox_type<return_tag_list> outbox =
        db::create<db::get_items<return_tag_list>>(
            // clang-tidy: std::move of trivially-copyable type
            std::move(mesh),  // NOLINT
            std::move(output_vars));
    return std::make_tuple(std::move(outbox));
  }
};

template <typename ParallelComponentOfReceiver>
struct SendNumElements {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static void apply(const db::DataBox<DbTags>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponentOfReceiver>(cache);

    Parallel::simple_action<ah::Actions::DataInterpolator::ReceiveNumElements>(
        *receiver_proxy.ckLocalBranch());
  }
};

template <typename ParallelComponentOfReceiver>
struct BeginHorizonSearch {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::list_contains_v<DbTags, Tags::Mesh<3>>> = nullptr>
  static void apply(const db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const std::vector<Time>& timesteps) noexcept {
    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponentOfReceiver>(cache);

    const ElementId<3> element_id{array_index};
    const auto& vars = db::get<ah::vars_tags_from_element>(box);
    const auto& mesh = db::get<Tags::Mesh<3>>(box);

    for (const auto& timestep : timesteps) {
      Parallel::simple_action<
          ah::Actions::DataInterpolator::GetVolumeDataFromElement>(
          *receiver_proxy.ckLocalBranch(), timestep, element_id, mesh, vars);
    }
  }
};

}  // namespace DgElementArray
}  // namespace Actions
}  // namespace test_ah
