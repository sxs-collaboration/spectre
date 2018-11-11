// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

namespace hydro {
// @{
/// Computes the Lorentz factor \f$W=1/\sqrt{1 - v^i v_i}\f$
template <typename DataType, size_t Dim, typename Fr>
Scalar<DataType> lorentz_factor(
    const tnsr::I<DataType, Dim, Fr>& spatial_velocity,
    const tnsr::i<DataType, Dim, Fr>& spatial_velocity_form) noexcept;

template <typename DataType>
Scalar<DataType> lorentz_factor(
    const Scalar<DataType>& spatial_velocity_squared) noexcept;
// @}

namespace Tags {
/// Compute item for Lorentz factor \f$W\f$.
///
/// Can be retrieved using `hydro::Tags::LorentzFactor`
template <typename DataType, size_t Dim, typename Fr>
struct LorentzFactorCompute : LorentzFactor<DataType>, db::ComputeTag {
  static constexpr auto function = &lorentz_factor<DataType, Dim, Fr>;
  using argument_tags = tmpl::list<SpatialVelocity<DataType, Dim, Fr>,
                                   SpatialVelocityOneForm<DataType, Dim, Fr>>;
};
}  // namespace Tags
}  // namespace hydro
