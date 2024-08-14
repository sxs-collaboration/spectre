// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TMPL.hpp"

class DataVector;

namespace ylm::Tags {
/// Defines type aliases used in Strahlkorper-related Tags.
namespace aliases {
template <typename Fr>
using Jacobian =
    Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
           index_list<SpatialIndex<3, UpLo::Up, Fr>,
                      SpatialIndex<2, UpLo::Lo, ::Frame::Spherical<Fr>>>>;
template <typename Fr>
using InvJacobian =
    Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
           index_list<SpatialIndex<2, UpLo::Up, ::Frame::Spherical<Fr>>,
                      SpatialIndex<3, UpLo::Lo, Fr>>>;
template <typename Fr>
using InvHessian = Tensor<
    DataVector, tmpl::integral_list<std::int32_t, 3, 2, 1>,
    index_list<SpatialIndex<2, UpLo::Up, ::Frame::Spherical<Fr>>,
               SpatialIndex<3, UpLo::Lo, Fr>, SpatialIndex<3, UpLo::Lo, Fr>>>;
}  // namespace aliases
}  // namespace ylm::Tags
