// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \ingroup GeneralRelativityGroup
/// Holds functions related to general relativity.
namespace gr {
/// @{
/*!
 * \brief Computes spacetime normal one-form from lapse.
 *
 * \details If \f$\alpha\f$ is the lapse, then
 *
 * \f{align}
 *     n_t &= - \alpha \\
 *     n_i &= 0
 * \f}
 *
 * is computed.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
void spacetime_normal_one_form(
    gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> normal_one_form,
    const Scalar<DataType>& lapse);

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::a<DataType, SpatialDim, Frame> spacetime_normal_one_form(
    const Scalar<DataType>& lapse);
/// @}

namespace Tags {
/*!
 * \brief Compute item for spacetime normal oneform \f$n_a\f$ from
 * the lapse \f$\alpha\f$.
 *
 * \details Can be retrieved using `gr::Tags::SpacetimeNormalOneForm`.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
struct SpacetimeNormalOneFormCompute
    : SpacetimeNormalOneForm<DataType, SpatialDim, Frame>,
      db::ComputeTag {
  using argument_tags = tmpl::list<Lapse<DataType>>;

  using return_type = tnsr::a<DataType, SpatialDim, Frame>;

  static constexpr auto function =
      static_cast<void (*)(gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*>,
                           const Scalar<DataType>&)>(
          &spacetime_normal_one_form<DataType, SpatialDim, Frame>);

  using base = SpacetimeNormalOneForm<DataType, SpatialDim, Frame>;
};
}  // namespace Tags
}  // namespace gr
