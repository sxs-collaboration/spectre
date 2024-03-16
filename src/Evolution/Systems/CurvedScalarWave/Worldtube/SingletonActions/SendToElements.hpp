// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Inboxes.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct TimeStepId;
}  // namespace Tags
/// \endcond

namespace CurvedScalarWave::Worldtube::Actions {
/*!
 * \brief Sends the regular field coefficients to each element abutting the
 * worldtube.
 */
template <typename Metavariables>
struct SendToElements {
  static constexpr size_t Dim = Metavariables::volume_dim;
  using psi_tag = CurvedScalarWave::Tags::Psi;
  using dt_psi_tag = ::Tags::dt<CurvedScalarWave::Tags::Psi>;
  using tags_to_send = tmpl::list<psi_tag, dt_psi_tag>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const size_t order = db::get<Tags::ExpansionOrder>(box);
    auto& element_proxies = Parallel::get_parallel_component<
        typename Metavariables::dg_element_array>(cache);
    const auto& faces_grid_coords =
        get<Tags::ElementFacesGridCoordinates<Dim>>(box);
    const auto& psi_l0 =
        get<Stf::Tags::StfTensor<Tags::PsiWorldtube, 0, Dim, Frame::Inertial>>(
            box);
    const auto& dt_psi_l0 =
        get<Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>, 0, Dim,
                                 Frame::Inertial>>(box);
    const auto& psi_l1 =
        get<Stf::Tags::StfTensor<Tags::PsiWorldtube, 1, Dim, Frame::Inertial>>(
            box);
    const auto& dt_psi_l1 =
        get<Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>, 1, Dim,
                                 Frame::Inertial>>(box);
    const size_t num_coefs = order == 0 ? 1 : 4;
    Variables<tags_to_send> vars_to_send(num_coefs);
    get(get<psi_tag>(vars_to_send))[0] = get(psi_l0);
    get(get<dt_psi_tag>(vars_to_send))[0] = get(dt_psi_l0);
    if (order > 0) {
      for (size_t i = 0; i < Dim; ++i) {
        get(get<psi_tag>(vars_to_send))[i + 1] = psi_l1.get(i);
        get(get<dt_psi_tag>(vars_to_send))[i + 1] = dt_psi_l1.get(i);
      }
    }
    for (const auto& [element_id, _] : faces_grid_coords) {
      auto vars_to_send_copy = vars_to_send;
      Parallel::receive_data<Tags::RegularFieldInbox<Dim>>(
          element_proxies[element_id], db::get<::Tags::TimeStepId>(box),
          std::move(vars_to_send_copy));
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace CurvedScalarWave::Worldtube::Actions
