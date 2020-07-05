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
//@{
/*!
 * \brief Computes spacetime normal one-form from lapse.
 *
 * \details If \f$N\f$ is the lapse, then
 * \f{align} n_t &= - N \\
 * n_i &= 0 \f}
 * is computed.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_normal_one_form(
    gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> normal_one_form,
    const Scalar<DataType>& lapse) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> spacetime_normal_one_form(
    const Scalar<DataType>& lapse) noexcept;
//@}

namespace Tags {
/*!
 * \brief Compute item for spacetime normal oneform \f$n_a\f$ from
 * the lapse \f$N\f$.
 *
 * \details Can be retrieved using `gr::Tags::SpacetimeNormalOneForm`.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
struct SpacetimeNormalOneFormCompute
    : SpacetimeNormalOneForm<SpatialDim, Frame, DataType>,
      db::ComputeTag {
  using argument_tags = tmpl::list<Lapse<DataType>>;

  using return_type = tnsr::a<DataType, SpatialDim, Frame>;

  static constexpr auto function =
      static_cast<void (*)(gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*>,
                           const Scalar<DataType>&) noexcept>(
          &spacetime_normal_one_form<SpatialDim, Frame, DataType>);

  using base = SpacetimeNormalOneForm<SpatialDim, Frame, DataType>;
};
}  // namespace Tags
}  // namespace gr
