// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace hydro {
/// @{
/// Computes the Lorentz factor \f$W=1/\sqrt{1 - v^i v_i}\f$
template <typename DataType, size_t Dim, typename Frame>
void lorentz_factor(
    gsl::not_null<Scalar<DataType>*> result,
    const tnsr::I<DataType, Dim, Frame>& spatial_velocity,
    const tnsr::i<DataType, Dim, Frame>& spatial_velocity_form) noexcept;

template <typename DataType, size_t Dim, typename Frame>
Scalar<DataType> lorentz_factor(
    const tnsr::I<DataType, Dim, Frame>& spatial_velocity,
    const tnsr::i<DataType, Dim, Frame>& spatial_velocity_form) noexcept;

template <typename DataType>
void lorentz_factor(gsl::not_null<Scalar<DataType>*> result,
                    const Scalar<DataType>& spatial_velocity_squared) noexcept;

template <typename DataType>
Scalar<DataType> lorentz_factor(
    const Scalar<DataType>& spatial_velocity_squared) noexcept;
/// @}

namespace Tags {
/// Compute item for Lorentz factor \f$W\f$.
///
/// Can be retrieved using `hydro::Tags::LorentzFactor`
template <typename DataType, size_t Dim, typename Frame>
struct LorentzFactorCompute : LorentzFactor<DataType>, db::ComputeTag {
  using argument_tags =
      tmpl::list<SpatialVelocity<DataType, Dim, Frame>,
                 SpatialVelocityOneForm<DataType, Dim, Frame>>;

  using return_type = Scalar<DataType>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataType>*>, const tnsr::I<DataType, Dim, Frame>&,
      const tnsr::i<DataType, Dim, Frame>&) noexcept>(
      &lorentz_factor<DataType, Dim, Frame>);

  using base = LorentzFactor<DataType>;
};
}  // namespace Tags
}  // namespace hydro
