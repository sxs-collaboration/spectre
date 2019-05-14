// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/Equations.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {

/*!
 * The set of Bondi quantities computed by hypersurface step, in the required
 * order of computation
 */
using bondi_hypersurface_step_tags =
    tmpl::list<Tags::BondiBeta, Tags::BondiQ, Tags::BondiU, Tags::BondiW,
               Tags::BondiH>;

/*!
 * Metafunction that is a `tmpl::list` of the temporary tags taken by the
 * `ComputeBondiIntegrand` computational struct.
 */
template <typename Tag>
using integrand_temporary_tags =
    typename ComputeBondiIntegrand<Tag>::temporary_tags;

/*!
 * \brief A type list for the set of `BoundaryValue` tags needed as an input to
 * any of the template specializations of
 *
 * `PrecomputeCceDependencies`.
 * \details This is provided for easy and maintainable
 * construction of a `Variables` or \ref DataBoxGroup with all of the quantities
 * necessary for a Cce computation or portion thereof.
 * A container of these tags should have size
 * `Spectral::Swsh::number_of_swsh_collocation_points(l_max)`.
 */
template <template <typename> class BoundaryPrefix>
using pre_computation_boundary_tags =
    tmpl::list<BoundaryPrefix<Tags::BondiR>,
               BoundaryPrefix<Tags::DuRDividedByR>>;

/*!
 * \brief A type list for the set of tags computed by the set of
 * template specializations of `PrecomputeCceDepedencies`.
 *
 * \details This is provided for
 * easy and maintainable construction of a `Variables` or `DataBox` with all of
 * the quantities needed for a Cce computation or component. The data structures
 * represented by these tags should each have size `number_of_radial_points *
 * number_of_swsh_collocation_points(l_max)`. All of these tags may be computed
 * at once if using a \ref DataBoxGroup using the template
 * `mutate_all_precompute_cce_dependencies` or individually using
 * the template specializations `PrecomputeCceDependencies`.
 */
using pre_computation_tags =
    tmpl::list<Tags::DuRDividedByR, Tags::EthRDividedByR,
               Tags::EthEthRDividedByR, Tags::EthEthbarRDividedByR,
               Tags::BondiK, Tags::OneMinusY, Tags::BondiR>;

}  // namespace Cce
