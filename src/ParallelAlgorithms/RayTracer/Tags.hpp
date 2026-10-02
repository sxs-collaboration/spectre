// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

namespace ray_tracing::Tags {

/*!
 * Coordinate position of the ray
 */
template <typename DataType, size_t Dim = 3, typename Frame = Frame::Inertial>
struct Position : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};

/*!
 * Spatial momentum variable $\Pi_i = \frac{p_i}{\alpha p^0} =
 * \frac{p_i}{\sqrt{\gamma^{jk} p_j p_k}}$ as defined in \cite Bohn:2014xxa
 * (Eq. (3)) or \cite Bohn:2016afc (Eq. (5)).
 * See `gr::geodesic_equation` for details.
 */
template <typename DataType, size_t Dim = 3, typename Frame = Frame::Inertial>
struct Momentum : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

/*!
 * Redshift variable \f$\ln(\alpha p^0)\f$ as defined in \cite Bohn:2014xxa
 * (Eq. (5)).
 * See `gr::geodesic_equation` for details.
 */
template <typename DataType>
struct Redshift : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * Coordinate time to integrate the ray.
 */
struct IntegrationTime : db::SimpleTag {
  using type = double;
};

}  // namespace ray_tracing::Tags
