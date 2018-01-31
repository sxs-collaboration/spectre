// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Wedge2D.hpp"
#include "Domain/CoordinateMaps/Wedge3D.hpp"
#include "Parallel/CharmPupable.hpp"

namespace DomainCreators {
namespace DomainCreators_detail {
template <size_t Dim>
void register_with_charm();

template <>
void register_with_charm<1>() {
  PUPable_reg(SINGLE_ARG(::CoordinateMap<Frame::Logical, Frame::Inertial,
                                         CoordinateMaps::Affine>));
}

template <>
void register_with_charm<2>() {
  PUPable_reg(SINGLE_ARG(
      ::CoordinateMap<Frame::Logical, Frame::Inertial,
                      CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                                     CoordinateMaps::Affine>>));
  PUPable_reg(
      SINGLE_ARG(::CoordinateMap<
                 Frame::Logical, Frame::Inertial,
                 CoordinateMaps::ProductOf2Maps<CoordinateMaps::Equiangular,
                                                CoordinateMaps::Equiangular>>));
  PUPable_reg(SINGLE_ARG(::CoordinateMap<Frame::Logical, Frame::Inertial,
                                         CoordinateMaps::Wedge2D>));
}
template <>
void register_with_charm<3>() {
  PUPable_reg(SINGLE_ARG(
      ::CoordinateMap<Frame::Logical, Frame::Inertial,
                      CoordinateMaps::ProductOf3Maps<CoordinateMaps::Affine,
                                                     CoordinateMaps::Affine,
                                                     CoordinateMaps::Affine>>));
  PUPable_reg(
      SINGLE_ARG(::CoordinateMap<
                 Frame::Logical, Frame::Inertial,
                 CoordinateMaps::ProductOf3Maps<CoordinateMaps::Equiangular,
                                                CoordinateMaps::Equiangular,
                                                CoordinateMaps::Equiangular>>));

  PUPable_reg(SINGLE_ARG(::CoordinateMap<Frame::Logical, Frame::Inertial,
                                         CoordinateMaps::Wedge3D>));
}
}  // namespace DomainCreators_detail

void register_derived_with_charm() {
  DomainCreators_detail::register_with_charm<1>();
  DomainCreators_detail::register_with_charm<2>();
  DomainCreators_detail::register_with_charm<3>();
}
}  // namespace DomainCreators
