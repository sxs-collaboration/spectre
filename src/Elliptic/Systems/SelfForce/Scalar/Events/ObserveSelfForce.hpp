// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
#include "Elliptic/Tags.hpp"
#include "IO/Observer/GetSectionObservationKey.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/TypeTraits.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "PointwiseFunctions/AnalyticData/SelfForce/Scalar/CircularOrbit.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/OptionalHelpers.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarSelfForce::Events {

namespace detail {
tnsr::i<std::complex<double>, 2> extract_self_force(
    const tnsr::I<double, 2, Frame::ElementLogical>& puncture_logical_coords,
    const AnalyticData::CircularOrbit& circular_orbit,
    const Scalar<ComplexDataVector>& field, const Mesh<2>& mesh,
    const InverseJacobian<DataVector, 2, Frame::ElementLogical,
                          Frame::Inertial>& inv_jacobian);
}  // namespace detail

/*!
 * \brief Observe the self-force at the position of the scalar charge.
 *
 * \details This event interpolates the scalar m-mode field and its derivatives
 * to the position of the scalar charge (puncture) and computes the m-mode
 * contribution to the self-force acting on the charge. The self-force is then
 * written to a reduction file.
 *
 * The self-force is computed in the Boyer-Lindquist `r` and `theta`
 * coordinates (for scalar charge $q=1$) following Eq. (3.10) in
 * \cite Osburn:2022bby :
 * \begin{align}
 *   F_r^m &= \partial_r (\Psi_m^R / r) = \frac{1}{r \alpha}
 *     \partial_{r_*} \Psi_m^R - \frac{1}{r^2} \Psi_m^R \\
 *   F_\theta^m &= \partial_\theta (\Psi_m^R / r) = -\frac{1}{r}
 *     \partial_{\cos\theta} \Psi_m^R
 * \end{align}
 * with $\alpha = 1 - 2 M r / (r^2 + a^2)$. For modes $m > 0$ the self-force
 * includes an additional factor of 2 and a complex rotation by
 * $m \Delta\phi = \frac{m a}{r_+ - r_-}\ln\frac{r - r_+}{r - r_-}$.
 *
 * These contributions can be summed over all m-modes (in postprocessing) to
 * form the total self-force $F_\mu^\mathrm{self} = \sum_m F_\mu^m$, where the
 * sum is truncated at some maximum m-mode number.
 */
template <typename ArraySectionIdTag = void>
class ObserveSelfForce : public Event {
 public:
  static constexpr Options::String help =
      "Observe the self-force at the position of the scalar charge.";
  using options = tmpl::list<>;

  ObserveSelfForce() = default;

  explicit ObserveSelfForce(CkMigrateMessage* msg) : Event(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveSelfForce);  // NOLINT

  using compute_tags_for_observation_box = tmpl::list<>;

  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<::Tags::ObservationBox>;

  using BackgroundTag =
      elliptic::Tags::Background<elliptic::analytic_data::Background>;

  template <typename ComputeTagsList, typename DataBoxType,
            typename Metavariables, typename ParallelComponent>
  void operator()(const ObservationBox<ComputeTagsList, DataBoxType>& box,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ElementId<2>& element_id,
                  const ParallelComponent* const /*meta*/,
                  const ObservationValue& observation_value) const {
    // Skip observation on elements that are not part of the section
    const std::optional<std::string> section_observation_key =
        observers::get_section_observation_key<ArraySectionIdTag>(box);
    if (not section_observation_key.has_value()) {
      return;
    }
    const auto& background = get<BackgroundTag>(box);
    const auto& circular_orbit =
        dynamic_cast<const AnalyticData::CircularOrbit&>(background);
    // Get element-logical coords of puncture
    const auto& domain = get<domain::Tags::Domain<2>>(box);
    const auto puncture_position = circular_orbit.puncture_position();
    const auto& block = domain.blocks()[element_id.block_id()];
    const auto block_logical_coords =
        block_logical_coordinates_single_point(puncture_position, block);
    if (not block_logical_coords.has_value()) {
      return;
    }
    const auto puncture_logical_coords =
        element_logical_coordinates(block_logical_coords.value(), element_id);
    if (not puncture_logical_coords.has_value()) {
      return;
    }
    // Extract self-force
    const auto& field = get<Tags::MMode>(box);
    const auto& mesh = get<domain::Tags::Mesh<2>>(box);
    const auto& inv_jacobian =
        get<domain::Tags::InverseJacobian<2, Frame::ElementLogical,
                                          Frame::Inertial>>(box);
    const auto self_force =
        detail::extract_self_force(puncture_logical_coords.value(),
                                   circular_orbit, field, mesh, inv_jacobian);
    // Write result to file
    auto& reduction_writer = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(cache);
    Parallel::threaded_action<
        observers::ThreadedActions::WriteReductionDataRow>(
        reduction_writer[0], std::string{"/SelfForce"},
        std::vector<std::string>{"IterationId", "NumberOfGridPoints",
                                 "Re(SelfForce_r)", "Im(SelfForce_r)",
                                 "Re(SelfForce_theta)", "Im(SelfForce_theta)"},
        std::make_tuple(observation_value.value, mesh.number_of_grid_points(),
                        get<0>(self_force).real(), get<0>(self_force).imag(),
                        get<1>(self_force).real(), get<1>(self_force).imag()));
  }

  using observation_registration_tags = tmpl::list<::Tags::DataBox>;

  template <typename DbTagsList>
  std::optional<
      std::pair<observers::TypeOfObservation, observers::ObservationKey>>
  get_observation_type_and_key_for_registration(
      const db::DataBox<DbTagsList>& box) const {
    const std::optional<std::string> section_observation_key =
        observers::get_section_observation_key<ArraySectionIdTag>(box);
    if (not section_observation_key.has_value()) {
      return std::nullopt;
    }
    return {
        {observers::TypeOfObservation::Reduction,
         observers::ObservationKey("ObserveSelfForce" +
                                   section_observation_key.value() + ".dat")}};
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return false; }
};

/// \cond
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
template <typename ArraySectionIdTag>
PUP::able::PUP_ID ObserveSelfForce<ArraySectionIdTag>::my_PUP_ID = 0;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
/// \endcond
}  // namespace ScalarSelfForce::Events
