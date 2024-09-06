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
#include "Domain/ExcisionSphere.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/ElementActions/ReceiveWorldtubeData.hpp"
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

/// \cond
namespace Tags {
struct TimeStepId;
}  // namespace Tags
/// \endcond

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
 *    - `Worldtube::Tags::FaceCoordinates<Dim, Frame::Inertial, true>`
 *    - `Worldtube::Tags::PunctureField`
 *    - `Worldtube::Tags::ExcisionSphere`
 *    - `Tags::TimeStepId`
 */
struct SendToWorldtube {
  static constexpr size_t Dim = 3;
  using tags_to_send = tmpl::list<CurvedScalarWave::Tags::Psi,
                                  ::Tags::dt<CurvedScalarWave::Tags::Psi>>;
  using tags_to_slice_to_face =
      tmpl::list<CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi,
                 CurvedScalarWave::Tags::Phi<Dim>,
                 gr::Tags::Shift<DataVector, Dim>, gr::Tags::Lapse<DataVector>,
                 domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                               Frame::Inertial>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& element_id = db::get<domain::Tags::Element<Dim>>(box).id();
    const auto& excision_sphere = db::get<Tags::ExcisionSphere<Dim>>(box);
    const auto direction = excision_sphere.abutting_direction(element_id);
    if (not direction.has_value()) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const auto face_mesh = mesh.slice_away(direction->dimension());
    const size_t face_size = face_mesh.number_of_grid_points();

    Variables<tmpl::list<
        CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
        ::Tags::TempScalar<0>, ::Tags::TempScalar<1>, ::Tags::TempScalar<2>>>
        temporaries(face_size);
    auto& psi_regular_times_det =
        get(get<CurvedScalarWave::Tags::Psi>(temporaries));
    auto& dt_psi_regular_times_det =
        get(get<::Tags::dt<CurvedScalarWave::Tags::Psi>>(temporaries));
    auto& theta = get(get<::Tags::TempScalar<0>>(temporaries));
    auto& phi = get(get<::Tags::TempScalar<1>>(temporaries));
    auto& spherical_harmonic = get(get<::Tags::TempScalar<2>>(temporaries));
    const auto& face_quantities = db::get<Tags::FaceQuantities>(box).value();
    const auto& psi_numerical_face =
        get<CurvedScalarWave::Tags::Psi>(face_quantities);
    const auto& dt_psi_numerical_face =
        get<::Tags::dt<CurvedScalarWave::Tags::Psi>>(face_quantities);
    const auto& area_element =
        get<gr::surfaces::Tags::AreaElement<DataVector>>(face_quantities);
    const auto& puncture_field =
        db::get<Tags::CurrentIteration>(box) > 0
            ? db::get<Tags::IteratedPunctureField<Dim>>(box).value()
            : db::get<Tags::PunctureField<Dim>>(box).value();
    const auto& psi_puncture = get<CurvedScalarWave::Tags::Psi>(puncture_field);
    const auto& dt_psi_puncture =
        get<::Tags::dt<CurvedScalarWave::Tags::Psi>>(puncture_field);

    psi_regular_times_det =
        (get(psi_numerical_face) - get(psi_puncture)) * get(area_element);
    dt_psi_regular_times_det =
        (get(dt_psi_numerical_face) - get(dt_psi_puncture)) * get(area_element);
    const auto& centered_face_coords =
        db::get<Tags::FaceCoordinates<Dim, Frame::Inertial, true>>(box);
    ASSERT(centered_face_coords.has_value(),
           "Should be an abutting element here, but face coords are not "
           "calculated!");
    const auto& x = get<0>(centered_face_coords.value());
    const auto& y = get<1>(centered_face_coords.value());
    const auto& z = get<2>(centered_face_coords.value());

    const size_t order = db::get<Worldtube::Tags::ExpansionOrder>(box);
    const size_t num_modes = (order + 1) * (order + 1);
    Variables<tags_to_send> Ylm_coefs(num_modes);
    theta = atan2(hypot(x, y), z);
    phi = atan2(y, x);
    size_t index = 0;
    // project onto spherical harmonics
    for (size_t l = 0; l <= order; ++l) {
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      for (int m = -l; m <= static_cast<int>(l); ++m, ++index) {
        spherical_harmonic = ylm::real_spherical_harmonic(theta, phi, l, m);
        get(get<CurvedScalarWave::Tags::Psi>(Ylm_coefs)).at(index) =
            definite_integral(psi_regular_times_det * spherical_harmonic,
                              face_mesh);
        get(get<::Tags::dt<CurvedScalarWave::Tags::Psi>>(Ylm_coefs)).at(index) =
            definite_integral(dt_psi_regular_times_det * spherical_harmonic,
                              face_mesh);
      }
    }
    ASSERT(index == num_modes, "Internal indexing error. "
                                   << num_modes
                                   << " modes should have been calculated but "
                                   << index << " modes were computed.");

    auto& worldtube_component = Parallel::get_parallel_component<
        Worldtube::WorldtubeSingleton<Metavariables>>(cache);
    Parallel::receive_data<Worldtube::Tags::SphericalHarmonicsInbox<Dim>>(
        worldtube_component, db::get<::Tags::TimeStepId>(box),
        std::make_pair(element_id, std::move(Ylm_coefs)));
    if (db::get<Tags::CurrentIteration>(box) + 1 <
        db::get<Tags::MaxIterations>(box) ) {
      db::mutate<Tags::CurrentIteration>(
          [](const gsl::not_null<size_t*> current_iteration) {
            *current_iteration += 1;
          },
          make_not_null(&box));
      // still iterating, go to `IteratePunctureField`
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};

    } else {
      db::mutate<Tags::CurrentIteration>(
          [](const gsl::not_null<size_t*> current_iteration) {
            *current_iteration = 0;
          },
          make_not_null(&box));
      // done iterating, get data for BCs
      return {Parallel::AlgorithmExecution::Continue,
              tmpl::index_of<ActionList, ReceiveWorldtubeData>::value};
    }
  }
};
}  // namespace CurvedScalarWave::Worldtube::Actions
