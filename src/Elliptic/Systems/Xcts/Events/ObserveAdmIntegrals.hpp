// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "IO/Observer/GetSectionObservationKey.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/TypeTraits.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

namespace Events {

/// @{
/*!
 * \brief Computes the ADM integrals locally (within one element).
 *
 * To get the total ADM integrals, the results need to be summed over in a
 * reduction.
 *
 * See `Xcts::adm_linear_momentum_surface_integrand` for details on the formula
 * for each integrand.
 */
void local_adm_integrals(
    gsl::not_null<Scalar<double>*> adm_mass,
    gsl::not_null<tnsr::I<double, 3>*> adm_linear_momentum,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3>& deriv_conformal_factor,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
    const tnsr::ii<DataVector, 3>& spatial_metric,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const tnsr::ii<DataVector, 3>& extrinsic_curvature,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inv_jacobian,
    const Mesh<3>& mesh, const Element<3>& element,
    const DirectionMap<3, tnsr::i<DataVector, 3>>& conformal_face_normals);
/// @}

/// @{
/*!
 * \brief Observe ADM integrals after the XCTS solve.
 *
 * The surface integrals are taken over the outer boundary, which is defined as
 * the domain boundary in the upper logical zeta direction.
 *
 * Writes reduction quantities:
 * - `Linear momentum`
 */
class ObserveAdmIntegrals : public Event {
 private:
  using ReductionData = Parallel::ReductionData<
      // ADM Mass
      Parallel::ReductionDatum<double, funcl::Plus<>>,
      // ADM Linear Momentum (x-component)
      Parallel::ReductionDatum<double, funcl::Plus<>>,
      // ADM Linear Momentum (y-component)
      Parallel::ReductionDatum<double, funcl::Plus<>>,
      // ADM Linear Momentum (z-component)
      Parallel::ReductionDatum<double, funcl::Plus<>>>;

 public:
  /// \cond
  explicit ObserveAdmIntegrals(CkMigrateMessage* msg) : Event(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveAdmIntegrals);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr Options::String help =
      "Observe ADM integrals after the XCTS solve.\n"
      "\n"
      "Writes reduction quantities:\n"
      "- Linear momentum";

  ObserveAdmIntegrals() = default;

  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<ReductionData>>;

  using compute_tags_for_observation_box = tmpl::list<>;

  using return_tags = tmpl::list<>;

  using argument_tags = tmpl::list<
      Xcts::Tags::ConformalFactor<DataVector>,
      ::Tags::deriv<Xcts::Tags::ConformalFactor<DataVector>, tmpl::size_t<3>,
                    Frame::Inertial>,
      Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
      Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
      Xcts::Tags::ConformalChristoffelSecondKind<DataVector, 3,
                                                 Frame::Inertial>,
      Xcts::Tags::ConformalChristoffelContracted<DataVector, 3,
                                                 Frame::Inertial>,
      gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>,
      gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>,
      gr::Tags::ExtrinsicCurvature<DataVector, 3, Frame::Inertial>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      domain::Tags::InverseJacobian<3, Frame::ElementLogical, Frame::Inertial>,
      domain::Tags::Mesh<3>, domain::Tags::Element<3>,
      domain::Tags::Faces<3, domain::Tags::FaceNormal<3>>>;

  template <typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  void operator()(
      const Scalar<DataVector>& conformal_factor,
      const tnsr::i<DataVector, 3>& deriv_conformal_factor,
      const tnsr::ii<DataVector, 3>& conformal_metric,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
      const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
      const tnsr::ii<DataVector, 3>& spatial_metric,
      const tnsr::II<DataVector, 3>& inv_spatial_metric,
      const tnsr::ii<DataVector, 3>& extrinsic_curvature,
      const Scalar<DataVector>& trace_extrinsic_curvature,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      const Mesh<3>& mesh, const Element<3>& element,
      const DirectionMap<3, tnsr::i<DataVector, 3>>& conformal_face_normals,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ParallelComponent* const /*meta*/,
      const ObservationValue& observation_value) const {
    Scalar<double> adm_mass;
    tnsr::I<double, 3> adm_linear_momentum;
    local_adm_integrals(
        make_not_null(&adm_mass), make_not_null(&adm_linear_momentum),
        conformal_factor, deriv_conformal_factor, conformal_metric,
        inv_conformal_metric, conformal_christoffel_second_kind,
        conformal_christoffel_contracted, spatial_metric, inv_spatial_metric,
        extrinsic_curvature, trace_extrinsic_curvature, inv_jacobian, mesh,
        element, conformal_face_normals);

    // Save components of linear momentum as reduction data
    ReductionData reduction_data{get(adm_mass), get<0>(adm_linear_momentum),
                                 get<1>(adm_linear_momentum),
                                 get<2>(adm_linear_momentum)};
    std::vector<std::string> legend{"AdmMass", "AdmLinearMomentum_x",
                                    "AdmLinearMomentum_y",
                                    "AdmLinearMomentum_z"};

    // Get information required for reduction
    auto& local_observer = *Parallel::local_branch(
        Parallel::get_parallel_component<
            tmpl::conditional_t<Parallel::is_nodegroup_v<ParallelComponent>,
                                observers::ObserverWriter<Metavariables>,
                                observers::Observer<Metavariables>>>(cache));
    observers::ObservationId observation_id{observation_value.value,
                                            subfile_path_ + ".dat"};
    Parallel::ArrayComponentId array_component_id{
        std::add_pointer_t<ParallelComponent>{nullptr},
        Parallel::ArrayIndex<ElementId<3>>(array_index)};

    // Send reduction action
    if constexpr (Parallel::is_nodegroup_v<ParallelComponent>) {
      Parallel::threaded_action<
          observers::ThreadedActions::CollectReductionDataOnNode>(
          local_observer, std::move(observation_id),
          std::move(array_component_id), subfile_path_, std::move(legend),
          std::move(reduction_data));
    } else {
      Parallel::simple_action<observers::Actions::ContributeReductionData>(
          local_observer, std::move(observation_id),
          std::move(array_component_id), subfile_path_, std::move(legend),
          std::move(reduction_data));
    }
  }

  using observation_registration_tags = tmpl::list<>;

  std::optional<
      std::pair<observers::TypeOfObservation, observers::ObservationKey>>
  get_observation_type_and_key_for_registration() const {
    return {{observers::TypeOfObservation::Reduction,
             observers::ObservationKey(subfile_path_ + ".dat")}};
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return false; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    Event::pup(p);
    p | subfile_path_;
  }

 private:
  std::string subfile_path_ = "/AdmIntegrals";
};
/// @}

}  // namespace Events
