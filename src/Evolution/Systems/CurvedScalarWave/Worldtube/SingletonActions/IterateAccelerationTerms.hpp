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
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SelfForce.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/ReceiveElementData.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube::Actions {
/*!
 * \brief Computes the next iteration of the acceleration due to scalar self
 * force from the current iteration of the regular field.
 */
template <typename Metavariables>
struct IterateAccelerationTerms {
  static constexpr size_t Dim = Metavariables::volume_dim;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    ASSERT(db::get<Tags::MaxIterations>(box) > 1,
           "Internal action error: In Action `IterateAccelerationTerms` but "
           "`MaxIterations` is less than 2.");
    const size_t data_size = 3;
    Scalar<DataVector> data_to_send(data_size);
    const auto& geodesic_acc = db::get<Tags::GeodesicAcceleration<Dim>>(box);
    const auto& dt_psi_monopole =
        db::get<Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>, 0, Dim,
                                     Frame::Inertial>>(box);
    const auto& psi_dipole = db::get<
        Stf::Tags::StfTensor<Tags::PsiWorldtube, 1, Dim, Frame::Inertial>>(box);
    const auto& particle_velocity =
        db::get<Tags::ParticlePositionVelocity<Dim>>(box).at(1);
    const auto& background = get<Tags::BackgroundQuantities<Dim>>(box);
    const auto& inverse_metric =
        get<gr::Tags::InverseSpacetimeMetric<double, Dim>>(background);
    const auto& dilation_factor = get<Tags::TimeDilationFactor>(background);
    const auto self_force_acc = self_force_acceleration(
        dt_psi_monopole, psi_dipole, particle_velocity,
        db::get<Tags::Charge>(box), db::get<Tags::Mass>(box).value(),
        inverse_metric, dilation_factor);
    for (size_t i = 0; i < Dim; ++i) {
      get(data_to_send)[i] = geodesic_acc.get(i) + self_force_acc.get(i);
    }
    const auto& faces_grid_coords =
        get<Tags::ElementFacesGridCoordinates<Dim>>(box);
    auto& element_proxies = Parallel::get_parallel_component<
        typename Metavariables::dg_element_array>(cache);
    for (const auto& [element_id, _] : faces_grid_coords) {
      auto data_to_send_copy = data_to_send;
      Parallel::receive_data<Tags::SelfForceInbox<Dim>>(
          element_proxies[element_id], db::get<::Tags::TimeStepId>(box),
          std::move(data_to_send_copy));
    }
    return {Parallel::AlgorithmExecution::Continue,
            tmpl::index_of<ActionList, ReceiveElementData>::value};
  }
};
}  // namespace CurvedScalarWave::Worldtube::Actions
