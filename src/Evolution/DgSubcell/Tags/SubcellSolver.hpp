// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/SpatialDiscretization/OptionTags.hpp"
#include "Options/Options.hpp"

namespace evolution::dg::subcell::OptionTags {
/*!
 * \brief Group holding options for controlling the subcell solver
 * discretization.
 *
 * For example, this would hold the reconstruction scheme or order of the finite
 * difference derivatives.
 *
 * \note The `SubcellSolverGroup` is a subgroup of
 * `SpatialDiscretization::OptionTags::SpatialDiscretizationGroup`.
 */
struct SubcellSolverGroup {
  static std::string name() { return "SubcellSolver"; }
  static constexpr Options::String help{
      "Options controlling the subcell solver spatial discretization "
      "of the PDE system.\n\n"
      "Contains options such as what reconstruction scheme to use or what "
      "order of finite difference derivatives to apply."};
  using group = SpatialDiscretization::OptionTags::SpatialDiscretizationGroup;
};
}  // namespace evolution::dg::subcell::OptionTags
