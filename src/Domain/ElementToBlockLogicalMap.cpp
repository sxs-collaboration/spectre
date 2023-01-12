// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/ElementToBlockLogicalMap.hpp"

#include <cstddef>
#include <memory>

#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Structure/Side.hpp"

namespace domain {

using AffineMap = CoordinateMaps::Affine;

template <>
std::unique_ptr<
    CoordinateMapBase<Frame::ElementLogical, Frame::BlockLogical, 1>>
element_to_block_logical_map(const ElementId<1>& element_id) {
  return std::make_unique<
      CoordinateMap<Frame::ElementLogical, Frame::BlockLogical, AffineMap>>(
      AffineMap{-1., 1., element_id.segment_id(0).endpoint(Side::Lower),
                element_id.segment_id(0).endpoint(Side::Upper)});
}
template <>
std::unique_ptr<
    CoordinateMapBase<Frame::ElementLogical, Frame::BlockLogical, 2>>
element_to_block_logical_map(const ElementId<2>& element_id) {
  using AffineMap2D = CoordinateMaps::ProductOf2Maps<AffineMap, AffineMap>;
  return std::make_unique<
      CoordinateMap<Frame::ElementLogical, Frame::BlockLogical, AffineMap2D>>(
      AffineMap2D{{-1., 1., element_id.segment_id(0).endpoint(Side::Lower),
                   element_id.segment_id(0).endpoint(Side::Upper)},
                  {-1., 1., element_id.segment_id(1).endpoint(Side::Lower),
                   element_id.segment_id(1).endpoint(Side::Upper)}});
}

template <>
std::unique_ptr<
    CoordinateMapBase<Frame::ElementLogical, Frame::BlockLogical, 3>>
element_to_block_logical_map(const ElementId<3>& element_id) {
  using AffineMap3D =
      CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
  return std::make_unique<
      CoordinateMap<Frame::ElementLogical, Frame::BlockLogical, AffineMap3D>>(
      AffineMap3D{{-1., 1., element_id.segment_id(0).endpoint(Side::Lower),
                   element_id.segment_id(0).endpoint(Side::Upper)},
                  {-1., 1., element_id.segment_id(1).endpoint(Side::Lower),
                   element_id.segment_id(1).endpoint(Side::Upper)},
                  {-1., 1., element_id.segment_id(2).endpoint(Side::Lower),
                   element_id.segment_id(2).endpoint(Side::Upper)}});
}

}  // namespace domain
