// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

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


}  // namespace Cce
