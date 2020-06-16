// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

/// \cond
namespace domain {
namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace Tags
}  // namespace domain
class DataVector;
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

namespace GeneralizedHarmonic {
namespace Tags {
/*!
 * \brief Compute items to compute constraint-damping parameters for a
 * single-BH evolution.
 *
 * \details Can be retrieved using
 * `GeneralizedHarmonic::Tags::ConstraintGamma0`,
 * `GeneralizedHarmonic::Tags::ConstraintGamma1`, and
 * `GeneralizedHarmonic::Tags::ConstraintGamma2`.
 */
template <size_t SpatialDim, typename Frame>
struct ConstraintGamma0Compute : ConstraintGamma0, db::ComputeTag {
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<SpatialDim, Frame>>;

  using return_type = Scalar<DataVector>;

  static constexpr void function(
      const gsl::not_null<Scalar<DataVector>*> gamma,
      const tnsr::I<DataVector, SpatialDim, Frame>& coords) noexcept {
    destructive_resize_components(gamma, get<0>(coords).size());
    get(*gamma) =
        3. * exp(-0.0078125 * get(dot_product(coords, coords))) + 0.001;
  }

  using base = ConstraintGamma0;
};
/// \copydoc ConstraintGamma0Compute
template <size_t SpatialDim, typename Frame>
struct ConstraintGamma1Compute : ConstraintGamma1, db::ComputeTag {
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<SpatialDim, Frame>>;

  using return_type = Scalar<DataVector>;

  static constexpr void function(
      const gsl::not_null<Scalar<DataVector>*> gamma1,
      const tnsr::I<DataVector, SpatialDim, Frame>& coords) noexcept {
    destructive_resize_components(gamma1, get<0>(coords).size());
    get(*gamma1) = -1.;
  }

  using base = ConstraintGamma1;
};
/// \copydoc ConstraintGamma0Compute
template <size_t SpatialDim, typename Frame>
struct ConstraintGamma2Compute : ConstraintGamma2, db::ComputeTag {
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<SpatialDim, Frame>>;

  using return_type = Scalar<DataVector>;

  static constexpr void function(
      const gsl::not_null<Scalar<DataVector>*> gamma,
      const tnsr::I<DataVector, SpatialDim, Frame>& coords) noexcept {
    destructive_resize_components(gamma, get<0>(coords).size());
    get(*gamma) = exp(-0.0078125 * get(dot_product(coords, coords))) + 0.001;
  }

  using base = ConstraintGamma2;
};
}  // namespace Tags
}  // namespace GeneralizedHarmonic
