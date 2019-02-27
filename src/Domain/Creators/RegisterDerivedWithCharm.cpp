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
// alias to CoordinateMaps namespace:
namespace Maps = CoordinateMaps;

// aliases to individual CoordinateMaps:
using Affine = Maps::Affine;
using Affine2D = Maps::ProductOf2Maps<Affine, Affine>;
using Affine3D = Maps::ProductOf3Maps<Affine, Affine, Affine>;
using Equiangular = Maps::Equiangular;
using Equiangular2D = Maps::ProductOf2Maps<Equiangular, Equiangular>;
using Equiangular3D =
    Maps::ProductOf3Maps<Equiangular, Equiangular, Equiangular>;
using Equiangular3DPrism =
    Maps::ProductOf3Maps<Equiangular, Equiangular, Affine>;
using EquatorialCompression = Maps::EquatorialCompression;
using Frustum = Maps::Frustum;
using Wedge2D = Maps::Wedge2D;
using Wedge3D = Maps::Wedge3D;
using Wedge3DPrism = Maps::ProductOf2Maps<Wedge2D, Affine>;
using Identity2D = Maps::Identity<2>;
using Translation3D = Maps::ProductOf2Maps<Affine, Identity2D>;
using DiscreteRotation1D = Maps::DiscreteRotation<1>;
using DiscreteRotation2D = Maps::DiscreteRotation<2>;
using DiscreteRotation3D = Maps::DiscreteRotation<3>;

template <typename... Maps>
using LGMap = CoordinateMap<Frame::Logical, Frame::Grid, Maps...>;

template <typename... Maps>
using LIMap = CoordinateMap<Frame::Logical, Frame::Inertial, Maps...>;

template <size_t Dim>
void register_with_charm();

template <>
void register_with_charm<1>() {
  PUPable_reg(SINGLE_ARG(LIMap<Affine>));
  PUPable_reg(SINGLE_ARG(LIMap<DiscreteRotation1D, Affine>));
}

template <>
void register_with_charm<2>() {
  PUPable_reg(SINGLE_ARG(LIMap<Affine2D>));
  PUPable_reg(SINGLE_ARG(LIMap<DiscreteRotation2D, Affine2D>));
  PUPable_reg(SINGLE_ARG(LIMap<Equiangular2D>));
  PUPable_reg(SINGLE_ARG(LIMap<Identity2D>));
  PUPable_reg(SINGLE_ARG(LIMap<Wedge2D>));
}
template <>
void register_with_charm<3>() {
  PUPable_reg(SINGLE_ARG(LGMap<Affine3D>));
  PUPable_reg(SINGLE_ARG(LIMap<Affine3D>));
  PUPable_reg(SINGLE_ARG(LIMap<DiscreteRotation3D, Affine3D>));
  PUPable_reg(SINGLE_ARG(LGMap<Equiangular3D>));
  PUPable_reg(SINGLE_ARG(LIMap<Equiangular3D>));
  PUPable_reg(SINGLE_ARG(LIMap<Equiangular3DPrism>));
  PUPable_reg(SINGLE_ARG(LGMap<Frustum>));
  PUPable_reg(SINGLE_ARG(LIMap<Frustum>));
  PUPable_reg(SINGLE_ARG(LGMap<Wedge3D>));
  PUPable_reg(SINGLE_ARG(LIMap<Wedge3D>));
  PUPable_reg(SINGLE_ARG(LIMap<Wedge3D, EquatorialCompression>));
  PUPable_reg(SINGLE_ARG(LIMap<Wedge3DPrism>));
  PUPable_reg(SINGLE_ARG(LIMap<Wedge3D, EquatorialCompression, Translation3D>));
}
}  // namespace DomainCreators_detail

void register_derived_with_charm() {
  DomainCreators_detail::register_with_charm<1>();
  DomainCreators_detail::register_with_charm<2>();
  DomainCreators_detail::register_with_charm<3>();
}
}  // namespace creators
}  // namespace domain
