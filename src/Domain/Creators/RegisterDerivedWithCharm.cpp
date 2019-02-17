// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/EquatorialCompression.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/Frustum.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Wedge2D.hpp"
#include "Domain/CoordinateMaps/Wedge3D.hpp"

namespace domain {
namespace creators {
namespace DomainCreators_detail {
using Affine = CoordinateMaps::Affine;
using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
using Equiangular = CoordinateMaps::Equiangular;
using Equiangular2D = CoordinateMaps::ProductOf2Maps<Equiangular, Equiangular>;
using Equiangular3D =
    CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Equiangular>;
using Equiangular3DPrism =
    CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Affine>;
using EquatorialCompression = CoordinateMaps::EquatorialCompression;
using Wedge2D = CoordinateMaps::Wedge2D;
using Wedge3D = CoordinateMaps::Wedge3D;
using Wedge3DPrism = CoordinateMaps::ProductOf2Maps<Wedge2D, Affine>;
using Identity2D = CoordinateMaps::Identity<2>;
using Translation3D = CoordinateMaps::ProductOf2Maps<Affine, Identity2D>;

template <size_t Dim>
void register_with_charm();

template <>
void register_with_charm<1>() {
  PUPable_reg(
      SINGLE_ARG(CoordinateMap<Frame::Logical, Frame::Inertial, Affine>));
  PUPable_reg(
      SINGLE_ARG(CoordinateMap<Frame::Logical, Frame::Inertial,
                               CoordinateMaps::DiscreteRotation<1>, Affine>));
}

template <>
void register_with_charm<2>() {
  PUPable_reg(
      SINGLE_ARG(CoordinateMap<Frame::Logical, Frame::Inertial, Affine2D>));
  PUPable_reg(
      SINGLE_ARG(CoordinateMap<Frame::Logical, Frame::Inertial,
                               CoordinateMaps::DiscreteRotation<2>, Affine2D>));
  PUPable_reg(SINGLE_ARG(
      CoordinateMap<Frame::Logical, Frame::Inertial, Equiangular2D>));
  PUPable_reg(SINGLE_ARG(CoordinateMap<Frame::Logical, Frame::Inertial,
                                       CoordinateMaps::Identity<2>>));
  PUPable_reg(
      SINGLE_ARG(CoordinateMap<Frame::Logical, Frame::Inertial, Wedge2D>));
}
template <>
void register_with_charm<3>() {
  PUPable_reg(
      SINGLE_ARG(CoordinateMap<Frame::Logical, Frame::Inertial, Affine3D>));
  PUPable_reg(
      SINGLE_ARG(CoordinateMap<Frame::Logical, Frame::Inertial,
                               CoordinateMaps::DiscreteRotation<3>, Affine3D>));
  PUPable_reg(SINGLE_ARG(
      CoordinateMap<Frame::Logical, Frame::Inertial, Equiangular3D>));
  PUPable_reg(SINGLE_ARG(
      CoordinateMap<Frame::Logical, Frame::Inertial, Equiangular3DPrism>));
  PUPable_reg(SINGLE_ARG(
      CoordinateMap<Frame::Logical, Frame::Inertial, CoordinateMaps::Frustum>));
  PUPable_reg(
      SINGLE_ARG(CoordinateMap<Frame::Logical, Frame::Inertial, Wedge3D>));
  PUPable_reg(SINGLE_ARG(CoordinateMap<Frame::Logical, Frame::Inertial, Wedge3D,
                                       EquatorialCompression>));
  PUPable_reg(
      SINGLE_ARG(CoordinateMap<Frame::Logical, Frame::Inertial, Wedge3DPrism>));
  PUPable_reg(SINGLE_ARG(CoordinateMap<Frame::Logical, Frame::Inertial, Wedge3D,
                                       EquatorialCompression, Translation3D>));
}
}  // namespace DomainCreators_detail

void register_derived_with_charm() {
  DomainCreators_detail::register_with_charm<1>();
  DomainCreators_detail::register_with_charm<2>();
  DomainCreators_detail::register_with_charm<3>();
}
}  // namespace creators
}  // namespace domain
