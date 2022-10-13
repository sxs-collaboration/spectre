// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/PostInterpolationCallback.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace intrp {
namespace callbacks {

/* \brief post_interpolation_callback that outputs requested tensors
 * interpolated onto a LineSegment. The data is written as volume data into the
 * `Reductions` file with Quadrature `CellCentered` and Basis
 * `FiniteDifference`.
 *
 * Uses:
 * - Metavariables
 *   - `temporal_id`
 * - DataBox:
 *   - `TensorsToObserve`
 * - GlobalCache:
 *   - `observers::Tags::ReductionFileName`
 *
 * Conforms to the intrp::protocols::PostInterpolationCallback protocol
 *
 * For requirements on InterpolationTargetTag, see
 * intrp::protocols::InterpolationTargetTag
 *
 */
template <typename TensorsToObserve, typename InterpolationTargetTag>
struct ObserveLineSegment
    : tt::ConformsTo<intrp::protocols::PostInterpolationCallback> {
  static constexpr double fill_invalid_points_with =
      std::numeric_limits<double>::quiet_NaN();

  using const_global_cache_tags =
      tmpl::list<observers::Tags::ReductionFileName>;

  template <typename DbTags, typename Metavariables, typename TemporalId>
  static void apply(const db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const TemporalId& temporal_id) {
    static_assert(
        tmpl::list_contains_v<TensorsToObserve,
                              domain::Tags::Coordinates<
                                  Metavariables::volume_dim, Frame::Inertial>>,
        "When observing tensors on a line segment, please include the inertial "
        "coordinates in TensorsToObserve. This is so that the output file "
        "contains the coordinates, so then the output file is all that is "
        "needed for subsequent visualization or analysis of the output data.");
    std::vector<TensorComponent> tensor_components{};
    const size_t number_of_points =
        get<tmpl::front<TensorsToObserve>>(box)[0].size();
    tmpl::for_each<TensorsToObserve>(
        [&box, &tensor_components, &number_of_points](auto tag_v) {
          using Tag = tmpl::type_from<decltype(tag_v)>;
          const auto& tensor = get<Tag>(box);
          for (size_t i = 0; i < tensor.size(); ++i) {
            tensor_components.emplace_back(
                db::tag_name<Tag>() + tensor.component_suffix(i), tensor[i]);
            ASSERT(number_of_points == tensor[i].size(),
                   "All tensor components are expected to have the same size "
                       << number_of_points << ", but "
                       << db::tag_name<Tag>() + tensor.component_suffix(i)
                       << "has size " << tensor[i].size());
          }
        });

    const std::string& name = pretty_type::name<InterpolationTargetTag>();
    const std::string subfile_path{std::string{"/"} + name};
    const std::vector<size_t> extents_vector{number_of_points};
    const std::vector<Spectral::Basis> bases_vector{
        Spectral::Basis::FiniteDifference};
    const std::vector<Spectral::Quadrature> quadratures_vector{
        1, Spectral::Quadrature::CellCentered};
    const observers::ObservationId& observation_id = observers::ObservationId(
        InterpolationTarget_detail::get_temporal_id_value(temporal_id),
        subfile_path + ".vol");
    auto& proxy = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(cache);

    // We call this on proxy[0] because the 0th element of a NodeGroup is
    // always guaranteed to be present.
    Parallel::threaded_action<observers::ThreadedActions::WriteVolumeData>(
        proxy[0], Parallel::get<observers::Tags::ReductionFileName>(cache),
        subfile_path, observation_id,
        std::vector<ElementVolumeData>{{extents_vector, tensor_components,
                                        bases_vector, quadratures_vector,
                                        name}});
  }
};
}  // namespace callbacks
}  // namespace intrp
