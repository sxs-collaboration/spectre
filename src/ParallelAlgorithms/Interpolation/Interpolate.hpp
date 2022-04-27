// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <vector>

#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/AddTemporalIdsToInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorReceiveVolumeData.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t VolumeDim>
class ElementId;
namespace intrp {
template <typename Metavariables, typename Tag>
struct InterpolationTarget;
template <typename Metavariables>
struct Interpolator;
}  // namespace intrp
/// \endcond

namespace intrp {
template <typename InterpolationTargetTag, typename Tensors, size_t VolumeDim,
          typename Metavariables, typename... TensorTypes>
void interpolate(
    const typename InterpolationTargetTag::temporal_id::type& temporal_id,
    const Mesh<VolumeDim>& mesh, Parallel::GlobalCache<Metavariables>& cache,
    const ElementId<VolumeDim>& array_index, const TensorTypes&... tensors) {
  static_assert(
      std::is_same_v<
          tmpl::transform<Tensors, tmpl::bind<tmpl::type_from, tmpl::_1>>,
          tmpl::list<TensorTypes...>>,
      "Tensors passed do not match structures of Tensors list of tags.");
  Variables<Tensors> interp_vars(mesh.number_of_grid_points());
  const std::tuple<const TensorTypes&...> tensors_tuple{tensors...};
  tmpl::for_each<tmpl::make_sequence<tmpl::size_t<0>, sizeof...(TensorTypes)>>(
      [&interp_vars, &tensors_tuple](auto index_v) {
        constexpr size_t index = tmpl::type_from<decltype(index_v)>::value;
        get<tmpl::at_c<Tensors, index>>(interp_vars) =
            get<index>(tensors_tuple);
      });

  // Send volume data to the Interpolator, to trigger interpolation.
  auto& interpolator = *Parallel::local_branch(
      ::Parallel::get_parallel_component<Interpolator<Metavariables>>(cache));
  Parallel::simple_action<Actions::InterpolatorReceiveVolumeData<
      typename InterpolationTargetTag::temporal_id>>(
      interpolator, temporal_id, array_index, mesh, interp_vars);

  // Tell the interpolation target that it should interpolate.
  auto& target = Parallel::get_parallel_component<
      InterpolationTarget<Metavariables, InterpolationTargetTag>>(cache);
  Parallel::simple_action<
      Actions::AddTemporalIdsToInterpolationTarget<InterpolationTargetTag>>(
      target, std::vector<typename InterpolationTargetTag::temporal_id::type>{
                  temporal_id});
}
}  // namespace intrp
