// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <pup.h>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Domain/Tags/SurfaceJacobian.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbit.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Tags.hpp"
#include "Elliptic/Tags.hpp"
#include "IO/Observer/GetSectionObservationKey.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/TypeTraits.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace GrSelfForce::Events {

namespace detail {
std::pair<double, double> extract_flux(
    const AnalyticData::CircularOrbit& circular_orbit,
    const tnsr::aa<ComplexDataVector, 3>& field, const Mesh<2>& mesh,
    const Scalar<DataVector>& face_jacobian,
    const tnsr::I<DataVector, 2, Frame::Inertial>& face_coords);
}  // namespace detail

/*!
 * \brief Observe the outgoing energy flux at infinity.
 *
 * The time-averaged energy flux at infinity for a single $m$-mode is computed
 * as [Eq. (6.2) in \cite Osburn:2026uct ]:
 *
 * \begin{equation}
 * \langle \dot{E}^+_m \rangle = \frac{m^2 \Omega^2}{32} \int_0^\pi \left(
 *   |\psi^7_m|^2 + 4 |\psi^8_m|^2 + |\psi^9_m|^2
 *   - \psi^7_m \psi^{9*}_m - \psi^9_m \psi^{7*}_m
 * \right) \Big|_{r \to \infty} \sin\theta \, d\theta \text{,}
 * \end{equation}
 *
 * where $\Omega$ is the orbital frequency and $\psi^7_m$, $\psi^8_m$ and
 * $\psi^9_m$ are the $\theta\theta$, $\theta\phi$ and $\phi\phi$ components
 * of the field $\Psi_m$ (see `GrSelfForce::Tags::MMode`). The integrand is
 * evaluated in the equivalent form
 *
 * \begin{equation}
 * |\psi^7_m - \psi^9_m|^2 + 4 |\psi^8_m|^2 \text{,}
 * \end{equation}
 *
 * which avoids cancellation errors when $\psi^7_m \approx \psi^9_m$.
 *
 * As a consistency check of the angular integration, the event also reduces
 * the integral of the measure alone, $\int_0^\pi \sin\theta \, d\theta =
 * \int_{-1}^{1} d(\cos\theta)$, over the outer boundary. This is written to
 * the column "IntegralOfDCosTheta" and should equal 2.
 */
template <typename ArraySectionIdTag = void>
class ObserveFlux : public Event {
 public:
  using BackgroundTag =
      elliptic::Tags::Background<elliptic::analytic_data::Background>;

  using ReductionData = Parallel::ReductionData<
      // Observation value
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      // Number of grid points
      Parallel::ReductionDatum<size_t, funcl::Plus<>>,
      // Energy flux
      Parallel::ReductionDatum<double, funcl::Plus<>>,
      // Integral of d(cos(theta)) over the outer boundary (should be 2)
      Parallel::ReductionDatum<double, funcl::Plus<>>>;

  explicit ObserveFlux(CkMigrateMessage* msg) : Event(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveFlux);  // NOLINT

  using options = tmpl::list<>;

  static constexpr Options::String help =
      "Observe the energy flux at the outer boundary.";

  ObserveFlux() = default;

  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<ReductionData>>;
  using compute_tags_for_observation_box = tmpl::list<>;

  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<::Tags::ObservationBox>;

  template <typename ComputeTagsList, typename DataBoxType,
            typename Metavariables, typename ParallelComponent>
  void operator()(const ObservationBox<ComputeTagsList, DataBoxType>& box,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ElementId<2>& element_id,
                  const ParallelComponent* const /*meta*/,
                  const ObservationValue& observation_value) const {
    // Skip observation on elements that are not part of the section
    const std::optional<std::string> subfile_path = subfile_path_for_element(
        observers::get_section_observation_key<ArraySectionIdTag>(box));
    if (not subfile_path.has_value()) {
      return;
    }
    // Use only elements at the outer boundary
    const auto& direction = Direction<2>::upper_xi();
    const auto& element = get<domain::Tags::Element<2>>(box);
    const auto& mesh = get<domain::Tags::Mesh<2>>(box);
    double energy_flux = 0.;
    double integral_of_d_cos_theta = 0.;
    if (element.external_boundaries().contains(direction)) {
      const auto& background = get<BackgroundTag>(box);
      const auto& circular_orbit =
          dynamic_cast<const AnalyticData::CircularOrbit&>(background);
      std::tie(energy_flux, integral_of_d_cos_theta) = detail::extract_flux(
          circular_orbit, get<Tags::MMode>(box), mesh,
          get<domain::Tags::Faces<
              2, domain::Tags::DetSurfaceJacobian<Frame::ElementLogical,
                                                  Frame::Inertial>>>(box)
              .at(direction),
          get<domain::Tags::Faces<
              2, domain::Tags::Coordinates<2, Frame::Inertial>>>(box)
              .at(direction));
    }
    // Send data to reduction observer
    auto& local_observer = *Parallel::local_branch(
        Parallel::get_parallel_component<
            tmpl::conditional_t<Parallel::is_nodegroup_v<ParallelComponent>,
                                observers::ObserverWriter<Metavariables>,
                                observers::Observer<Metavariables>>>(cache));
    const std::string& subfile_path_with_suffix = subfile_path.value();
    observers::ObservationId observation_id{observation_value.value,
                                            subfile_path_with_suffix + ".dat"};
    Parallel::ArrayComponentId array_component_id{
        std::add_pointer_t<ParallelComponent>{nullptr},
        Parallel::ArrayIndex<ElementId<2>>(element_id)};
    ReductionData reduction_data{observation_value.value,
                                 mesh.number_of_grid_points(), energy_flux,
                                 integral_of_d_cos_theta};
    std::vector<std::string> legend{"ObservationValue", "NumberOfPoints",
                                    "EnergyFlux", "IntegralOfDCosTheta"};
    if constexpr (Parallel::is_nodegroup_v<ParallelComponent>) {
      Parallel::threaded_action<
          observers::ThreadedActions::CollectReductionDataOnNode>(
          local_observer, std::move(observation_id),
          std::move(array_component_id), subfile_path_with_suffix,
          std::move(legend), std::move(reduction_data));
    } else {
      Parallel::simple_action<observers::Actions::ContributeReductionData>(
          local_observer, std::move(observation_id),
          std::move(array_component_id), subfile_path_with_suffix,
          std::move(legend), std::move(reduction_data));
    }
  }

  using observation_registration_tags = tmpl::list<::Tags::DataBox>;

  template <typename DbTagsList>
  std::optional<
      std::pair<observers::TypeOfObservation, observers::ObservationKey>>
  get_observation_type_and_key_for_registration(
      const db::DataBox<DbTagsList>& box) const {
    const std::optional<std::string> subfile_path = subfile_path_for_element(
        observers::get_section_observation_key<ArraySectionIdTag>(box));
    if (not subfile_path.has_value()) {
      return std::nullopt;
    }
    return {{observers::TypeOfObservation::Reduction,
             observers::ObservationKey(subfile_path.value() + ".dat")}};
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return false; }

 private:
  // Subfile path (without ".dat") for elements in the section, or
  // `std::nullopt` for elements that are not part of the section. Used for
  // both registration and reduction so the observation keys always match.
  static std::optional<std::string> subfile_path_for_element(
      const std::optional<std::string>& section_observation_key) {
    if (not section_observation_key.has_value()) {
      return std::nullopt;
    }
    return "/Flux" + section_observation_key.value();
  }
};

/// \cond
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
template <typename ArraySectionIdTag>
PUP::able::PUP_ID ObserveFlux<ArraySectionIdTag>::my_PUP_ID = 0;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
/// \endcond
}  // namespace GrSelfForce::Events
