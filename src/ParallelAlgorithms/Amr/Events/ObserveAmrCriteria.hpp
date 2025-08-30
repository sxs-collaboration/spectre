// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/FloatingPointType.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Amr/Flag.hpp"
#include "IO/H5/TensorData.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Tags/Criteria.hpp"
#include "ParallelAlgorithms/Events/ObserveConstantsPerElement.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t VolumeDim>
class Domain;
template <size_t VolumeDim>
class ElementId;
namespace domain {
namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
namespace Tags {
template <size_t VolumeDim>
struct Domain;
struct FunctionsOfTime;
}  // namespace Tags
}  // namespace domain
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
struct Time;
struct TimeStep;
}  // namespace Tags
/// \endcond

namespace amr::Events {
namespace detail {
template <typename Criterion>
struct get_compute_tags {
  using type = typename Criterion::compute_tags_for_observation_box;
};
}  // namespace detail

/// \brief Observe the desired decisions of AMR criteria
///
/// \details The event will return a vector of decisions (an independent choice
/// in each logical dimension) for each of the AMR criteria.  These are the raw
/// choices made by each AMR critera, not taking into account any AMR policies.
/// Each element is output as a single cell with two points per dimension and
/// the observation constant on all those points.  The decisions are converted
/// to values as follows (in each logical dimension):
/// - -2.0 is for join with sibling (if possible)
/// - -1.0 is for decrease number of grid points
/// - 0.0 is for no change
/// - 1.0 is for increase number of grid points
/// - 2.0 is for splitting the element
template <typename Metavariables>
class ObserveAmrCriteria
    : public dg::Events::ObserveConstantsPerElement<Metavariables::volume_dim> {
 public:
  static constexpr size_t volume_dim = Metavariables::volume_dim;
  /// \cond
  explicit ObserveAmrCriteria(CkMigrateMessage* m)
      : dg::Events::ObserveConstantsPerElement<volume_dim>(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveAmrCriteria);  // NOLINT
  /// \endcond

  static constexpr Options::String help =
      "Observe the desired decisions of AMR criteria in the volume.";

  ObserveAmrCriteria() = default;

  ObserveAmrCriteria(const std::string& subfile_name,
                     ::FloatingPointType coordinates_floating_point_type,
                     ::FloatingPointType floating_point_type)
      : dg::Events::ObserveConstantsPerElement<volume_dim>(
            subfile_name, coordinates_floating_point_type,
            floating_point_type) {}

  using compute_tags_for_observation_box =
      tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
          tmpl::at<typename Metavariables::factory_creation::factory_classes,
                   Criterion>,
          detail::get_compute_tags<tmpl::_1>>>>;

  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<::Tags::ObservationBox>;

  template <typename DataBoxType, typename ComputeTagsList,
            typename ParallelComponent>
  void operator()(const ObservationBox<DataBoxType, ComputeTagsList>& box,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ElementId<Metavariables::volume_dim>& element_id,
                  const ParallelComponent* const component,
                  const Event::ObservationValue& observation_value) const {
    const auto& refinement_criteria = get<amr::Criteria::Tags::Criteria>(box);
    const double time = get<::Tags::Time>(box);
    const auto& functions_of_time = get<::domain::Tags::FunctionsOfTime>(box);
    const Domain<volume_dim>& domain =
        get<::domain::Tags::Domain<volume_dim>>(box);

    std::vector<TensorComponent> components = this->allocate_and_insert_coords(
        volume_dim * refinement_criteria.size(), time, functions_of_time,
        domain, element_id);

    for (const auto& criterion : refinement_criteria) {
      const auto decision = criterion->evaluate(box, cache, element_id);
      for (size_t d = 0; d < volume_dim; ++d) {
        this->add_constant(
            make_not_null(&components),
            criterion->observation_name() +
                tnsr::i<double, volume_dim,
                        Frame::ElementLogical>::component_suffix(d),
            static_cast<double>(gsl::at(decision, d)) - 3.0);
      }
    }

    this->observe(components, cache, element_id, component, observation_value);
  }

  bool needs_evolved_variables() const override { return true; }

  void pup(PUP::er& p) override {
    dg::Events::ObserveConstantsPerElement<volume_dim>::pup(p);
  }
};

/// \cond
#ifndef __CUDA_ARCH__
template <typename Metavariables>
PUP::able::PUP_ID ObserveAmrCriteria<Metavariables>::my_PUP_ID = 0;  // NOLINT
#endif  // __CUDA_ARCH__
/// \endcond
}  // namespace amr::Events
