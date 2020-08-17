// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"

namespace Cce {
namespace Frame {
/// The frame for the spherical metric in which the radial coordinate is an
/// affine parameter along outward-pointing null geodesics.
struct RadialNull {};
}  // namespace Frame

// tensor aliases for brevity
using SphericaliCartesianJ = Tensor<
    DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
    index_list<SpatialIndex<3, UpLo::Lo, ::Frame::Spherical<::Frame::Inertial>>,
               SpatialIndex<3, UpLo::Lo, ::Frame::Inertial>>>;

using CartesianiSphericalJ =
    Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
           index_list<SpatialIndex<3, UpLo::Lo, ::Frame::Inertial>,
                      SpatialIndex<3, UpLo::Up,
                                   ::Frame::Spherical<::Frame::Inertial>>>>;

using AngulariCartesianA = Tensor<
  DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
  index_list<SpatialIndex<2, UpLo::Lo, ::Frame::Spherical<::Frame::Inertial>>,
             SpacetimeIndex<3, UpLo::Lo, ::Frame::Inertial>>>;

using SphericaliCartesianjj = Tensor<
    DataVector, tmpl::integral_list<std::int32_t, 2, 1, 1>,
    index_list<SpatialIndex<3, UpLo::Lo, ::Frame::Spherical<::Frame::Inertial>>,
               SpatialIndex<3, UpLo::Lo, ::Frame::Inertial>,
               SpatialIndex<3, UpLo::Lo, ::Frame::Inertial>>>;

namespace Tags {
namespace detail {
// this provides a set of tags for the purposes of allocating once in the entire
// Boundary data computation; these tags are currently not used outside
// intermediate steps of the procedure in `BoundaryData.hpp`

struct CosPhi : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct CosTheta : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct SinPhi : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct SinTheta : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct CartesianCoordinates : db::SimpleTag {
  using type = tnsr::I<DataVector, 3>;
};

struct CartesianToSphericalJacobian : db::SimpleTag {
  using type = SphericaliCartesianJ;
};

struct InverseCartesianToSphericalJacobian : db::SimpleTag {
  using type = CartesianiSphericalJ;
};

struct WorldtubeNormal : db::SimpleTag {
  using type = tnsr::I<DataVector, 3>;
};

struct UpDyad : db::SimpleTag {
  using type = tnsr::I<ComplexDataVector, 2, Frame::RadialNull>;
};

struct DownDyad : db::SimpleTag {
  using type = tnsr::i<ComplexDataVector, 2, Frame::RadialNull>;
};

struct RealBondiR : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct AngularDNullL : db::SimpleTag {
  using type = AngulariCartesianA;
};

struct NullL : db::SimpleTag {
  using type = tnsr::A<DataVector, 3>;
};

template <typename Tag>
struct DLambda : db::SimpleTag {
  using type = typename Tag::type;
};

}  // namespace detail
}  // namespace Tags
}  // namespace Cce
