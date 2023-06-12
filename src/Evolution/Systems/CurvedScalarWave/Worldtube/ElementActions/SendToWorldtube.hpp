// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <cstddef>
#include <optional>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/AreaElement.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ExcisionSphere.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Inboxes.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonChare.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/RealSphericalHarmonics.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave::Worldtube::Actions {
/*!
 * \brief Projects the regular field \f$\Psi^R\f$ and its time derivative
 * \f$\partial_t \Psi^R\f$ onto real spherical harmonics and sends the result to
 * the worldtube.
 *
 * \details The regular field is obtained by subtracting the singular/puncture
 * field from the numerical DG field.
 * All spherical harmonics are computed for \f$l <= n\f$, where \f$n\f$ is the
 * worldtube expansion order. The projection is done by integrating over the DG
 * grid of the element face using \ref definite_integral with the euclidean area
 * element. The worldtube adds up all integrals from the different elements to
 * obtain the integral over the entire sphere.
 *
 * DataBox:
 * - Uses:
 *    - `tags_to_slice_on_face`
 *    - `Worldtube::Tags::ExpansionOrder`
 *    - `Worldtube::Tags::FaceCoordinates<Dim, Frame::Grid, true>`
 *    - `Worldtube::Tags::PunctureField`
 *    - `Worldtube::Tags::ExcisionSphere`
 *    - `Tags::TimeStepId`
 */
struct SendToWorldtube {
  static constexpr size_t Dim = 3;
  using tags_to_send = tmpl::list<CurvedScalarWave::Tags::Psi,
                                  ::Tags::dt<CurvedScalarWave::Tags::Psi>>;
  using tags_to_slice_to_face = tmpl::list<
      CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi,
      CurvedScalarWave::Tags::Phi<Dim>, gr::Tags::Shift<DataVector, Dim>,
      gr::Tags::Lapse<DataVector>,
      domain::Tags::InverseJacobian<Dim, Frame::ElementLogical, Frame::Grid>>;

  using inbox_tags = tmpl::list<Worldtube::Tags::SphericalHarmonicsInbox<Dim>>;
  using simple_tags = tmpl::list<Tags::RegularFieldAdvectiveTerm<Dim>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& puncture_field =
        db::get<Worldtube::Tags::PunctureField<Dim>>(box);
    if (puncture_field.has_value()) {
      const auto& element_id = db::get<domain::Tags::Element<Dim>>(box).id();
      const auto& excision_sphere = db::get<Tags::ExcisionSphere<Dim>>(box);
      const auto direction = excision_sphere.abutting_direction(element_id);
      ASSERT(direction.has_value(), "Should be abutting");
      const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
      const auto face_mesh = mesh.slice_away(direction->dimension());
      const size_t face_size = face_mesh.number_of_grid_points();

      Variables<tmpl::push_back<tags_to_slice_to_face, ::Tags::TempScalar<0>>>
          vars_on_face(face_size);

      tmpl::for_each<tags_to_slice_to_face>(
          [&box, &vars_on_face, &mesh, &direction](auto tag_to_slice_v) {
            using tag_to_slice = typename decltype(tag_to_slice_v)::type;
            data_on_slice(make_not_null(&get<tag_to_slice>(vars_on_face)),
                          db::get<tag_to_slice>(box), mesh.extents(),
                          direction.value().dimension(),
                          index_to_slice_at(mesh.extents(), direction.value()));
          });
      const auto& face_lapse = get<gr::Tags::Lapse<DataVector>>(vars_on_face);
      const auto& face_shift =
          get<gr::Tags::Shift<DataVector, Dim>>(vars_on_face);
      auto& face_inv_jacobian =
          get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                            Frame::Grid>>(vars_on_face);
      const auto& face_psi = get<CurvedScalarWave::Tags::Psi>(vars_on_face);
      const auto& face_pi = get<CurvedScalarWave::Tags::Pi>(vars_on_face);
      const auto& face_phi =
          get<CurvedScalarWave::Tags::Phi<Dim>>(vars_on_face);
      // re-use allocations
      Scalar<DataVector>& area_element =
          get<::Tags::TempScalar<0>>(vars_on_face);
      euclidean_area_element(make_not_null(&area_element), face_inv_jacobian,
                             direction.value());
      // re-use allocations
      DataVector& psi_regular_times_det = get<0, 0>(face_inv_jacobian);
      DataVector& dt_psi_regular_times_det = get<0, 1>(face_inv_jacobian);
      // the regular field is the full numerical field minus the puncture field
      psi_regular_times_det =
          get(face_psi) -
          get(get<CurvedScalarWave::Tags::Psi>(puncture_field.value()));

      // transform Pi to dt Psi. This is equivalent to the evolution equation
      // for dt Psi but we need to calculate it again because boundary
      // corrections from the DG scheme are already applied to the stored value.
      dt_psi_regular_times_det =
          -get(face_lapse) * get(face_pi) +
          get(dot_product(face_shift, face_phi)) -
          get(get<::Tags::dt<CurvedScalarWave::Tags::Psi>>(
              puncture_field.value()));

      const auto& mesh_velocity = db::get<domain::Tags::MeshVelocity<Dim>>(box);
      ASSERT(mesh_velocity.has_value(),
             "Expected a moving grid for worldrube evolution.");
      // is an optional so we can't put the tag into variables. The shift is not
      // used at this point so we use the allocation for the mesh_velocity to
      // save memory.
      auto& mesh_velocity_on_face =
          get<gr::Tags::Shift<DataVector, Dim>>(vars_on_face);
      data_on_slice(make_not_null(&mesh_velocity_on_face),
                    mesh_velocity.value(), mesh.extents(),
                    direction.value().dimension(),
                    index_to_slice_at(mesh.extents(), direction.value()));
      db::mutate<Tags::RegularFieldAdvectiveTerm<Dim>>(
          [&face_phi, &mesh_velocity_on_face,
           &di_psi_puncture =
               get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                                 Frame::Inertial>>(puncture_field.value())](
              const gsl::not_null<Scalar<DataVector>*> regular_advective_term) {
            tenex::evaluate<>(regular_advective_term,
                              (face_phi(ti::i) - di_psi_puncture(ti::i)) *
                                  mesh_velocity_on_face(ti::I));
          },
          make_not_null(&box));
      // The time derivative is transformed into the grid frame using the
      // advective term which comes from the transformation of the time
      // derivative due to the moving mesh.
      dt_psi_regular_times_det +=
          get(get<Tags::RegularFieldAdvectiveTerm<Dim>>(box));

      psi_regular_times_det *= get(area_element);
      dt_psi_regular_times_det *= get(area_element);
      const auto& centered_face_coords =
          db::get<Tags::FaceCoordinates<Dim, Frame::Grid, true>>(box);
      ASSERT(centered_face_coords.has_value(),
             "Should be an abutting element here, but face coords are not "
             "calculated!");
      const auto& x = get<0>(centered_face_coords.value());
      const auto& y = get<1>(centered_face_coords.value());
      const auto& z = get<2>(centered_face_coords.value());

      const size_t order = db::get<Worldtube::Tags::ExpansionOrder>(box);
      const size_t num_modes = (order + 1) * (order + 1);
      Variables<tags_to_send> Ylm_coefs(num_modes);
      // re-use allocations
      auto& theta = get<0, 2>(face_inv_jacobian);
      auto& phi = get<1, 0>(face_inv_jacobian);
      auto& spherical_harmonic = get<1, 1>(face_inv_jacobian);

      theta = atan2(hypot(x, y), z);
      phi = atan2(y, x);
      size_t index = 0;
      // project onto spherical harmonics
      for (size_t l = 0; l <= order; ++l) {
        for (int m = -l; m <= static_cast<int>(l); ++m, ++index) {
          spherical_harmonic = real_spherical_harmonic(theta, phi, l, m);
          get(get<CurvedScalarWave::Tags::Psi>(Ylm_coefs)).at(index) =
              definite_integral(psi_regular_times_det * spherical_harmonic,
                                face_mesh);
          get(get<::Tags::dt<CurvedScalarWave::Tags::Psi>>(Ylm_coefs))
              .at(index) = definite_integral(
              dt_psi_regular_times_det * spherical_harmonic, face_mesh);
        }
      }
      ASSERT(index == num_modes,
             "Internal indexing error. "
                 << num_modes << " modes should have been calculated but "
                 << index << " modes were computed.");

      auto& worldtube_component = Parallel::get_parallel_component<
          Worldtube::WorldtubeSingleton<Metavariables>>(cache);
      Parallel::receive_data<Worldtube::Tags::SphericalHarmonicsInbox<Dim>>(
          worldtube_component, db::get<::Tags::TimeStepId>(box),
          std::make_pair(element_id, std::move(Ylm_coefs)));
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace CurvedScalarWave::Worldtube::Actions
