// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/SpatialDiscretization/OptionTags.hpp"
#include "Options/Options.hpp"

namespace dg::OptionTags {
/*!
 * \brief Group holding options for controlling the DG discretization.
 *
 * For example, this would hold whether to use the strong or weak form, what
 * boundary correction/numerical flux to use, and which quadrature rule to use.
 *
 * \note The `DiscontinuousGalerkinGroup` is a subgroup of
 * `SpatialDiscretization::OptionTags::SpatialDiscretizationGroup`.
 */
struct DiscontinuousGalerkinGroup {
  static std::string name() noexcept { return "DiscontinuousGalerkin"; }
  static constexpr Options::String help{
      "Options controlling the discontinuous Galerkin spatial discretization "
      "of the PDE system.\n\n"
      "Contains options such as whether to use the strong or weak form, what "
      "boundary correction/numerical flud to use, and which quadrature rule to "
      "use."};
  using group = SpatialDiscretization::OptionTags::SpatialDiscretizationGroup;
};
}  // namespace dg::OptionTags
