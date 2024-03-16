// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Inboxes.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/PunctureField.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave::Worldtube::Actions {

/*!
 * \brief Checks if the regular field has been received from the worldtube and
 * computes the retarded field for boundary conditions.
 *
 *  \details This action checks whether the coefficients of Taylor Series of the
 * regular field have been sent by the worldtube. If so, the series is evaluated
 * at the face coordinate in the inertial frame  and the puncture field is added
 * to it to obtain the retarded field. This is stored in \ref
 * Tags::WorldtubeSolution which is used to formulate boundary conditions in
 * \ref CurvedScalarWave::BoundaryConditions::Worldtube.
 */
struct ReceiveWorldtubeData {
  static constexpr size_t Dim = 3;
  using psi_tag = CurvedScalarWave::Tags::Psi;
  using dt_psi_tag = ::Tags::dt<CurvedScalarWave::Tags::Psi>;
  template <typename Frame>
  using di_psi_tag =
      ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<Dim>, Frame>;
  using evolved_tags_list =
      typename CurvedScalarWave::System<Dim>::variables_tag::tags_list;
  using simple_tags = tmpl::list<Tags::WorldtubeSolution<Dim>>;
  using inbox_tags = tmpl::list<Tags::RegularFieldInbox<Dim>>;
  using tags_to_slice_to_face =
      tmpl::list<gr::Tags::Shift<DataVector, Dim>, gr::Tags::Lapse<DataVector>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& element_id = db::get<domain::Tags::Element<Dim>>(box).id();
    const auto& excision_sphere = db::get<Tags::ExcisionSphere<Dim>>(box);
    const auto direction = excision_sphere.abutting_direction(element_id);
    if (direction.has_value()) {
      const auto& time_step_id = db::get<::Tags::TimeStepId>(box);
      auto& inbox = get<Tags::RegularFieldInbox<Dim>>(inboxes);
      if (not inbox.count(time_step_id)) {
        return {Parallel::AlgorithmExecution::Retry, std::nullopt};
      }
      const auto& puncture_field =
          db::get<Tags::MaxIterations>(box) > 1
              ? db::get<Tags::IteratedPunctureField<Dim>>(box)
              : db::get<Tags::PunctureField<Dim>>(box);
      ASSERT(puncture_field.has_value(),
             "The puncture field should be initialized!");

      const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
      const auto face_mesh = mesh.slice_away(direction->dimension());
      const size_t face_size = face_mesh.number_of_grid_points();
      Variables<tags_to_slice_to_face> vars_on_face(face_size);
      tmpl::for_each<tags_to_slice_to_face>(
          [&box, &vars_on_face, &mesh, &direction](auto tag_to_slice_v) {
            using tag_to_slice = typename decltype(tag_to_slice_v)::type;
            data_on_slice(make_not_null(&get<tag_to_slice>(vars_on_face)),
                          db::get<tag_to_slice>(box), mesh.extents(),
                          direction.value().dimension(),
                          index_to_slice_at(mesh.extents(), direction.value()));
          });
      auto& received_data = inbox.at(time_step_id);
      const auto& centered_face_coords =
          db::get<Tags::FaceCoordinates<Dim, Frame::Inertial, true>>(box)
              .value();

      db::mutate<Tags::WorldtubeSolution<Dim>>(
          [&received_data, &puncture_field, &vars_on_face,
           &centered_face_coords,
           &expansion_order = db::get<Tags::ExpansionOrder>(box)](
              const gsl::not_null<Variables<evolved_tags_list>*>
                  worldtube_solution) {
            worldtube_solution->initialize(
                puncture_field.value().number_of_grid_points());

            auto& psi = get<psi_tag>(*worldtube_solution);
            auto& pi = get<CurvedScalarWave::Tags::Pi>(*worldtube_solution);
            auto& phi =
                get<CurvedScalarWave::Tags::Phi<Dim>>(*worldtube_solution);

            // the puncture field plus the monopole of the regular field
            get(psi) = get(get<psi_tag>(puncture_field.value())) +
                       get(get<psi_tag>(received_data))[0];
            get(pi) = get(get<dt_psi_tag>(puncture_field.value())) +
                      get(get<dt_psi_tag>(received_data))[0];
            for (size_t i = 0; i < Dim; ++i) {
              phi.get(i) =
                  get<di_psi_tag<Frame::Inertial>>(puncture_field.value())
                      .get(i);
            }
            if (expansion_order > 0) {
              // add on the dipole of the regular field
              for (size_t i = 0; i < Dim; ++i) {
                get(psi) += get(get<psi_tag>(received_data))[i + 1] *
                            centered_face_coords.get(i);
                get(pi) += get(get<dt_psi_tag>(received_data))[i + 1] *
                           centered_face_coords.get(i);
                phi.get(i) += get(get<psi_tag>(received_data))[i + 1];
              }
            }
            const auto& shift =
                get<gr::Tags::Shift<DataVector, Dim>>(vars_on_face);
            const auto& lapse = get<gr::Tags::Lapse<DataVector>>(vars_on_face);

            // convert dt_psi -> pi
            get(pi) *= -1.;
            get(pi) += get(dot_product(shift, phi));
            get(pi) /= get(lapse);
          },
          make_not_null(&box));
      inbox.erase(time_step_id);
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace CurvedScalarWave::Worldtube::Actions
