// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace hydro {

/// \brief Function computing the transport velocity.
///
/// Computes the transport velocity, using
/// \f$v_t^i=\alpha v^i-\beta^i\f$, with
/// $v^i$ being the spatial velocity, $\alpha$ the lapse, and
/// $\beta^i$ the shift.
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
void transport_velocity(gsl::not_null<tnsr::I<DataType, Dim, Fr>*> result,
                        const tnsr::I<DataType, Dim, Fr>& spatial_velocity,
                        const Scalar<DataType>& lapse,
                        const tnsr::I<DataType, Dim, Fr>& shift);

namespace Tags {
/// \brief Compute tag for the transport velocity.
///
/// Compute item for the transport velocity, using
/// \f$v_t^i=\alpha v^i-\beta^i\f$, with
/// $v^i$ being the spatial velocity, $\alpha$ the lapse, and
/// $\beta^i$ the shift.
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct TransportVelocityCompute
    : hydro::Tags::TransportVelocity<DataType, Dim, Fr>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<SpatialVelocity<DataType, Dim, Fr>,
                 ::gr::Tags::Lapse<DataType>,
                 ::gr::Tags::Shift<DataType, Dim, Fr>>;

  using base = hydro::Tags::TransportVelocity<DataType, Dim, Fr>;
  using return_type = typename base::type;

  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<tnsr::I<DataType, Dim, Fr>*> result,
      const tnsr::I<DataType, Dim, Fr>& spatial_velocity,
      const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Fr>& shift)>(
      &hydro::transport_velocity<DataType, Dim, Fr>);
};

}  // namespace Tags

}  // namespace hydro
