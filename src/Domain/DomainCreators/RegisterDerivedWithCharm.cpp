// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/AffineMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/DomainCreators/Interval.hpp"
#include "Parallel/CharmPupable.hpp"

namespace DomainCreators {
namespace DomainCreaters_detail {
template <size_t Dim>
void register_with_charm();

template <>
void register_with_charm<1>() {
  PUPable_reg(SINGLE_ARG(::CoordinateMap<Frame::Logical, Frame::Inertial,
                                         CoordinateMaps::AffineMap>));
}

template <>
void register_with_charm<2>() {
  PUPable_reg(
      SINGLE_ARG(::CoordinateMap<
                 Frame::Logical, Frame::Inertial,
                 CoordinateMaps::ProductOf2Maps<CoordinateMaps::AffineMap,
                                                CoordinateMaps::AffineMap>>));
}
template <>
void register_with_charm<3>() {
  PUPable_reg(SINGLE_ARG(
      ::CoordinateMap<Frame::Logical, Frame::Inertial,
                      CoordinateMaps::ProductOf3Maps<
                          CoordinateMaps::AffineMap, CoordinateMaps::AffineMap,
                          CoordinateMaps::AffineMap>>));
}
}  // namespace DomainCreaters_detail

void register_derived_with_charm() {
  DomainCreaters_detail::register_with_charm<1>();
  DomainCreaters_detail::register_with_charm<2>();
  DomainCreaters_detail::register_with_charm<3>();
}
}  // namespace DomainCreators
