// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependence/CompositionUniformTranslation.hpp"

#include <unordered_map>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/FunctionsOfTime/CombineFunctionsOfTime.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace domain::creators::time_dependence {

template <size_t MeshDim>
CompositionUniformTranslation<MeshDim>::CompositionUniformTranslation(
    std::unique_ptr<TimeDependence<MeshDim>> uniform_translation0,
    std::unique_ptr<TimeDependence<MeshDim>> uniform_translation1)
    : coord_map_(domain::push_back(
          dynamic_cast<UniformTranslation<MeshDim>&>(*uniform_translation0)
              .map_for_composition(),
          dynamic_cast<UniformTranslation<MeshDim>&>(*uniform_translation1)
              .map_for_composition())),
      functions_of_time_(domain::FunctionsOfTime::combine_functions_of_time(
          uniform_translation0->functions_of_time(),
          uniform_translation1->functions_of_time())) {}

template <size_t MeshDim>
CompositionUniformTranslation<MeshDim>::CompositionUniformTranslation(
    CoordMap coord_map,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time)
    : coord_map_(std::move(coord_map)) {
  functions_of_time_ = clone_unique_ptrs(functions_of_time);
}

template <size_t MeshDim>
auto CompositionUniformTranslation<MeshDim>::get_clone() const
    -> std::unique_ptr<TimeDependence<MeshDim>> {
  return std::make_unique<CompositionUniformTranslation>(coord_map_,
                                                         functions_of_time_);
}

template <size_t MeshDim>
auto CompositionUniformTranslation<MeshDim>::block_maps(size_t number_of_blocks)
    const -> std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, MeshDim>>> {
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, MeshDim>>>
      result{number_of_blocks};
  result[0] = std::make_unique<CoordMap>(coord_map_);
  for (size_t i = 1; i < number_of_blocks; ++i) {
    result[i] = result[0]->get_clone();
  }
  return result;
}

template <size_t MeshDim>
auto CompositionUniformTranslation<MeshDim>::functions_of_time() const
    -> std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> {
  return clone_unique_ptrs(functions_of_time_);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class CompositionUniformTranslation<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef DIM
#undef INSTANTIATION

}  // namespace domain::creators::time_dependence
