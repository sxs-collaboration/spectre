// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/ElementActions/SendToWorldtube.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Inboxes.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/PunctureField.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube::Actions {
/*!
 * \brief Computes an updated iteration of the puncture field given the current
 * acceleration of the charge sent by the worldtube singleton.
 */
struct IteratePunctureField {
  static constexpr size_t Dim = 3;

  using inbox_tags = tmpl::list<Tags::SelfForceInbox<Dim>>;
  using simple_tags = tmpl::list<Tags::IteratedPunctureField<Dim>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& centered_face_coordinates =
        db::get<Tags::FaceCoordinates<Dim, Frame::Inertial, true>>(box);
    if (not centered_face_coordinates.has_value()) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    const auto& time_step_id = db::get<::Tags::TimeStepId>(box);
    auto& inbox = get<Tags::SelfForceInbox<Dim>>(inboxes);
    if (not inbox.count(time_step_id)) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }
    db::mutate<Tags::IteratedPunctureField<Dim>>(
        [&self_force_data = get(inbox.at(time_step_id)),
         &position_velocity = db::get<Tags::ParticlePositionVelocity<Dim>>(box),
         &centered_face_coordinates, charge = db::get<Tags::Charge>(box),
         &puncture_field = db::get<Tags::PunctureFieldConfig>(box)](
            const auto iterated_puncture_field) {
          tnsr::I<double, Dim> iterated_acceleration{
              {self_force_data[0], self_force_data[1], self_force_data[2]}};
          const size_t face_size =
              get<0>(centered_face_coordinates.value()).size();

          if (not iterated_puncture_field->has_value()) {
            iterated_puncture_field->emplace(face_size);
          }

          puncture_field.apply_puncture(
              make_not_null(&iterated_puncture_field->value()),
              centered_face_coordinates.value(), position_velocity[0],
              position_velocity[1], iterated_acceleration);
          Variables<tmpl::list<CurvedScalarWave::Tags::Psi,
                               ::Tags::dt<CurvedScalarWave::Tags::Psi>,
                               ::Tags::deriv<CurvedScalarWave::Tags::Psi,
                                             tmpl::size_t<3>, Frame::Inertial>>>
              acceleration_terms(face_size);
          puncture_field.apply_acceleration_terms(
              make_not_null(&acceleration_terms),
              centered_face_coordinates.value(), position_velocity[0],
              position_velocity[1], iterated_acceleration, self_force_data[3],
              self_force_data[4], self_force_data[5], self_force_data[6],
              self_force_data[7], self_force_data[8], self_force_data[9],
              self_force_data[10], self_force_data[11], self_force_data[12],
              self_force_data[13], self_force_data[14]);
          iterated_puncture_field->value() += acceleration_terms;
          iterated_puncture_field->value() *= charge;
        },
        make_not_null(&box));
    inbox.erase(time_step_id);
    return {Parallel::AlgorithmExecution::Continue,
            tmpl::index_of<ActionList, SendToWorldtube>::value};
  }
};
}  // namespace CurvedScalarWave::Worldtube::Actions
