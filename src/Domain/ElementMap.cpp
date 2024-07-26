// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/ElementMap.hpp"

#include "Domain/CoordinateMaps/Composition.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/OptimizerHacks.hpp"

template <size_t Dim, typename TargetFrame>
ElementMap<Dim, TargetFrame>::ElementMap(
    const ElementId<Dim>& element_id,
    std::unique_ptr<
        domain::CoordinateMapBase<Frame::BlockLogical, TargetFrame, Dim>>
        block_map)
    : block_map_(std::move(block_map)),
      map_slope_{[](const ElementId<Dim>& id) {
        std::array<double, Dim> result{};
        for (size_t d = 0; d < Dim; ++d) {
          gsl::at(result, d) = 0.5 * (id.segment_id(d).endpoint(Side::Upper) -
                                      id.segment_id(d).endpoint(Side::Lower));
        }
        return result;
      }(element_id)},
      map_offset_{[](const ElementId<Dim>& id) {
        std::array<double, Dim> result{};
        for (size_t d = 0; d < Dim; ++d) {
          // The clang optimizer appears to generate code to execute
          // this loop extra times and then throw away the results.
          VARIABLE_CAUSES_CLANG_FPE(d);
          gsl::at(result, d) = 0.5 * (id.segment_id(d).endpoint(Side::Upper) +
                                      id.segment_id(d).endpoint(Side::Lower));
        }
        return result;
      }(element_id)} {}

// We could refactor the ElementMap class to use a `Composition` internally also
// for the element-to-block-logical map, so the whole map is just one
// composition.
template <size_t Dim, typename TargetFrame>
ElementMap<Dim, TargetFrame>::ElementMap(const ElementId<Dim>& element_id,
                                         const Block<Dim>& block)
    : ElementMap(
          element_id,
          [&block]() -> std::unique_ptr<domain::CoordinateMapBase<
                         Frame::BlockLogical, TargetFrame, Dim>> {
            if constexpr (std::is_same_v<TargetFrame, Frame::Inertial>) {
              if (block.is_time_dependent()) {
                using CompositionType = domain::CoordinateMaps::Composition<
                    tmpl::list<Frame::BlockLogical, Frame::Grid,
                               Frame::Inertial>,
                    Dim>;
                return std::make_unique<CompositionType>(
                    block.moving_mesh_logical_to_grid_map().get_clone(),
                    block.moving_mesh_grid_to_inertial_map().get_clone());
              } else {
                return block.stationary_map().get_clone();
              }
            } else if constexpr (std::is_same_v<TargetFrame, Frame::Grid>) {
              if (block.is_time_dependent()) {
                return block.moving_mesh_logical_to_grid_map().get_clone();
              } else {
                return block.stationary_map().get_to_grid_frame();
              }
            }
          }()) {
  ASSERT(element_id.block_id() == block.id(),
         "Element " << element_id << " is not in block " << block.id() << ".");
}

template <size_t Dim, typename TargetFrame>
void ElementMap<Dim, TargetFrame>::pup(PUP::er& p) {
  p | block_map_;
  p | map_slope_;
  p | map_offset_;
}

// For dual frame evolutions the ElementMap only goes to the grid frame
#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GET_FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(r, data) \
  template class ElementMap<GET_DIM(data), GET_FRAME(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3),
                        (Frame::Inertial, Frame::Grid))

#undef GET_DIM
#undef GET_FRAME
#undef INSTANTIATION
