// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Wedge2D.hpp"
#include "Domain/CoordinateMaps/Wedge3D.hpp"

namespace DomainCreators {
namespace DomainCreators_detail {
using Affine = CoordinateMaps::Affine;
using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
using Equiangular = CoordinateMaps::Equiangular;
using Equiangular2D = CoordinateMaps::ProductOf2Maps<Equiangular, Equiangular>;
using Equiangular3D =
    CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Equiangular>;

template <size_t Dim>
void register_with_charm();

template <>
void register_with_charm<1>() {
  PUPable_reg(
      SINGLE_ARG(::CoordinateMap<Frame::Logical, Frame::Inertial, Affine>));
}

template <>
void register_with_charm<2>() {
  PUPable_reg(
      SINGLE_ARG(::CoordinateMap<Frame::Logical, Frame::Inertial, Affine2D>));
  PUPable_reg(SINGLE_ARG(
      ::CoordinateMap<Frame::Logical, Frame::Inertial, Equiangular2D>));
  PUPable_reg(
      SINGLE_ARG(::CoordinateMap<Frame::Logical, Frame::Inertial, Affine2D,
                                 CoordinateMaps::DiscreteRotation<2>>));
  PUPable_reg(SINGLE_ARG(::CoordinateMap<Frame::Logical, Frame::Inertial,
                                         CoordinateMaps::Wedge2D>));
}
template <>
void register_with_charm<3>() {
  PUPable_reg(
      SINGLE_ARG(::CoordinateMap<Frame::Logical, Frame::Inertial, Affine3D>));
  PUPable_reg(SINGLE_ARG(
      ::CoordinateMap<Frame::Logical, Frame::Inertial, Equiangular3D>));

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
