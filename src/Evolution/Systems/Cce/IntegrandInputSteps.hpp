// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>

#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tags.hpp"
#include "Evolution/Systems/Cce/Equations.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
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

namespace detail {
// structs containing typelists for organizing the set of tags needed at each
// step of the CCE hypersurface evaluation steps. These are not directly the
// tags that are inputs in the `Equations.hpp`, as those would contain
// redundancy, and not necessarily obtain the derivatives during the most ideal
// steps. So, the ordering in these structs has been slightly 'designed' in ways
// that are not trivial to automate with tmpl list manipulation
template <typename Tag>
struct TagsToComputeForImpl;

template <>
struct TagsToComputeForImpl<Tags::BondiBeta> {
  using pre_swsh_derivative_tags =
      tmpl::list<Tags::Dy<Tags::BondiJ>, Tags::Dy<Tags::Dy<Tags::BondiJ>>>;
  using second_swsh_derivative_tags = tmpl::list<>;
  using swsh_derivative_tags = tmpl::list<>;
};

// Note: Due to the order in which Jacobians for the conversion between
// numerical and Bondi spin-weighted derivatives are evaluated, all of the
// higher (second) spin-weighted derivatives must be computed AFTER the
// eth(dy(bondi)) values (which act as inputs to the second derivative
// conversions to fixed Bondi radius), so must appear later in the
// `swsh_derivative_tags` typelists.
template <>
struct TagsToComputeForImpl<Tags::BondiQ> {
  using pre_swsh_derivative_tags =
      tmpl::list<Tags::Dy<Tags::BondiBeta>, Tags::Dy<Tags::Dy<Tags::BondiBeta>>,
                 ::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>,
                 ::Tags::Multiplies<Tags::BondiJbar, Tags::Dy<Tags::BondiJ>>>;
  using swsh_derivative_tags = tmpl::list<
      Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                       Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiBeta>,
                                       Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>,
          Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiJbar, Tags::Dy<Tags::BondiJ>>,
          Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiJ>,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                       Spectral::Swsh::Tags::Ethbar>>;
  using second_swsh_derivative_tags = tmpl::list<>;
};

template <>
struct TagsToComputeForImpl<Tags::BondiU> {
  using pre_swsh_derivative_tags =
      tmpl::list<Tags::Exp2Beta, Tags::Dy<Tags::BondiQ>,
                 Tags::Dy<Tags::Dy<Tags::BondiQ>>>;
  using second_swsh_derivative_tags = tmpl::list<>;
  using swsh_derivative_tags = tmpl::list<>;
};

template <>
struct TagsToComputeForImpl<Tags::BondiW> {
  using pre_swsh_derivative_tags =
      tmpl::list<Tags::Dy<Tags::BondiU>, Tags::Dy<Tags::Dy<Tags::BondiU>>,
                 Tags::Dy<::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>>>;
  using swsh_derivative_tags = tmpl::list<
      Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiBeta>,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<
          Tags::Dy<::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>>,
          Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<
          Tags::Dy<::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>>,
          Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiU>,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiU,
                                       Spectral::Swsh::Tags::Ethbar>>;
  // Currently, the `eth_ethbar_j` term is the single instance of a swsh
  // derivative needing nested `Spectral::Swsh::Tags::Derivatives` steps to
  // compute. The reason is that if we do not compute the derivative in two
  // steps, there are intermediate terms in the Jacobian which depend on eth_j,
  // which is a spin-weight 3 quantity and therefore cannot be computed with
  // libsharp (the SWSH library being used). If `eth_ethbar_j` becomes not
  // needed, the remaining `second_swsh_derivative_tags` can be merged to the
  // end of `swsh_derivative_tags` and the corresponding computational steps
  // from `SwshDerivatives.hpp` removed.
  using second_swsh_derivative_tags =
      tmpl::list<Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                                  Spectral::Swsh::Tags::EthEth>,
                 Spectral::Swsh::Tags::Derivative<
                     Tags::BondiBeta, Spectral::Swsh::Tags::EthEthbar>,
                 Spectral::Swsh::Tags::Derivative<
                     ::Tags::Multiplies<Tags::BondiJ, Tags::BondiJbar>,
                     Spectral::Swsh::Tags::EthEthbar>,

                 Spectral::Swsh::Tags::Derivative<
                     Tags::BondiJ, Spectral::Swsh::Tags::EthbarEthbar>,
                 Spectral::Swsh::Tags::Derivative<
                     Spectral::Swsh::Tags::Derivative<
                         Tags::BondiJ, Spectral::Swsh::Tags::Ethbar>,
                     Spectral::Swsh::Tags::Eth>>;
};

template <>
struct TagsToComputeForImpl<Tags::BondiH> {
  using pre_swsh_derivative_tags =
      tmpl::list<::Tags::Multiplies<Tags::BondiJbar, Tags::BondiU>,
                 ::Tags::Multiplies<Tags::BondiUbar, Tags::Dy<Tags::BondiJ>>,
                 Tags::JbarQMinus2EthBeta, Tags::Dy<Tags::BondiW>>;
  using swsh_derivative_tags = tmpl::list<
      Spectral::Swsh::Tags::Derivative<Tags::BondiQ, Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiU, Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiUbar, Tags::Dy<Tags::BondiJ>>,
          Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiJbar, Tags::Dy<Tags::BondiJ>>,
          Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Tags::JbarQMinus2EthBeta,
                                       Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<
          ::Tags::Multiplies<Tags::BondiJbar, Tags::BondiU>,
          Spectral::Swsh::Tags::Ethbar>,
      Spectral::Swsh::Tags::Derivative<Tags::BondiQ,
                                       Spectral::Swsh::Tags::Ethbar>>;
  using second_swsh_derivative_tags = tmpl::list<>;
};
}  // namespace detail

/*!
 * \brief A typelist for the set of `BoundaryValue` tags needed as an input to
 * any of the template specializations of `PrecomputeCceDependencies`.
 *
 * \details This is provided for easy and maintainable
 * construction of a `Variables` or \ref DataBoxGroup with all of the quantities
 * necessary for a CCE computation or portion thereof.
 * A container of these tags should have size
 * `Spectral::Swsh::number_of_swsh_collocation_points(l_max)`.
 */
template <template <typename> class BoundaryPrefix>
using pre_computation_boundary_tags =
    tmpl::list<BoundaryPrefix<Tags::BondiR>,
               BoundaryPrefix<Tags::DuRDividedByR>>;

/*!
 * \brief A typelist for the set of tags computed by the set of
 * template specializations of `PrecomputeCceDepedencies`.
 *
 * \details This is provided for easy and maintainable construction of a
 * `Variables` or \ref DataBoxGroup with all of the quantities needed for a CCE
 * computation or component. The data structures represented by these tags
 * should each have size `number_of_radial_points *
 * Spectral::Swsh::number_of_swsh_collocation_points(l_max)`. All of these tags
 * may be computed at once if using a \ref DataBoxGroup using the template
 * `mutate_all_precompute_cce_dependencies` or individually using
 * the template specializations `PrecomputeCceDependencies`.
 *
 * \note the tag `Tags::DuRDividedByR` is omitted from this list because in the
 * case where a gauge transformation must be applied, the time derivative
 * quantities must wait until later in the computation.
 */
using pre_computation_tags =
    tmpl::list<Tags::EthRDividedByR, Tags::EthEthRDividedByR,
               Tags::EthEthbarRDividedByR, Tags::BondiK, Tags::OneMinusY,
               Tags::BondiR>;

// @{
/*!
 * \brief A typelist for the set of tags computed by the set of
 * template specializations of `ComputePreSwshDerivatives`.
 *
 * \details This is provided for easy and maintainable construction of a
 * `Variables` or \ref DataBoxGroup with all of the quantities needed for a CCE
 * computation or component. The data structures represented by these tags
 * should each have size `number_of_radial_points *
 * Spectral::Swsh::number_of_swsh_collocation_points(l_max)`. All of these tags
 * (for a given integrated Bondi quantity) may be computed at once if using a
 * \ref DataBoxGroup using the template
 * `mutate_all_pre_swsh_derivatives_for_tag` or individually using the template
 * specializations of `ComputePreSwshDerivatives`. The full set of integrated
 * Bondi quantities is available from the typelist
 * `bondi_hypersurface_step_tags`.
 */
template <typename Tag>
struct pre_swsh_derivative_tags_to_compute_for {
  using type =
      typename detail::TagsToComputeForImpl<Tag>::pre_swsh_derivative_tags;
};

template <typename Tag>
using pre_swsh_derivative_tags_to_compute_for_t =
    typename pre_swsh_derivative_tags_to_compute_for<Tag>::type;
// @}

// @{
/*!
 * \brief A typelist for the set of tags computed by single spin-weighted
 * differentiation using utilities from the `Swsh` namespace.
 */
template <typename Tag>
struct single_swsh_derivative_tags_to_compute_for {
  using type = typename detail::TagsToComputeForImpl<Tag>::swsh_derivative_tags;
};

template <typename Tag>
using single_swsh_derivative_tags_to_compute_for_t =
    typename single_swsh_derivative_tags_to_compute_for<Tag>::type;
// @}

// @{
/*!
 * \brief A typelist for the set of tags computed by multiple spin-weighted
 * differentiation using utilities from the `Swsh` namespace.
 */
template <typename Tag>
struct second_swsh_derivative_tags_to_compute_for {
  using type =
      typename detail::TagsToComputeForImpl<Tag>::second_swsh_derivative_tags;
};

template <typename Tag>
using second_swsh_derivative_tags_to_compute_for_t =
    typename single_swsh_derivative_tags_to_compute_for<Tag>::type;
// @}

/*!
 * \brief A typelist for the set of tags computed by spin-weighted
 * differentiation using utilities from the `Swsh` namespace.
 *
 * \details This is provided for easy and maintainable construction of a
 * `Variables` or \ref DataBoxGroup with all of the quantities needed for a CCE
 * computation or component. The data structures represented by these tags
 * should each have size `number_of_radial_points *
 * Spectral::Swsh::number_of_swsh_collocation_points(l_max)`. All of these tags
 * (for a given integrated Bondi quantity) may be computed at once if using a
 * \ref DataBoxGroup using the template `mutate_all_swsh_derivatives_for_tag`.
 * Individual tag computation is not provided in a convenient interface, as
 * there is significant savings in aggregating spin-weighted differentiation
 * steps. The full set of integrated Bondi quantities is available from the
 * typelist `bondi_hypersurface_step_tags`.
 */
using all_swsh_derivative_tags =
    tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
        bondi_hypersurface_step_tags,
        tmpl::bind<tmpl::list,
                   single_swsh_derivative_tags_to_compute_for<tmpl::_1>,
                   second_swsh_derivative_tags_to_compute_for<tmpl::_1>>>>>;

/*!
 * \brief A typelist for the full set of coefficient buffers needed to process
 * all of the tags in `all_swsh_derivative_tags` using batch processing provided
 * in `mutate_all_swsh_derivatives_for_tag`.
 *
 * \details This is provided for easy and maintainable construction of a
 * `Variables` or \ref DataBoxGroup with all of the quantities needed for a CCE
 * computation or component. The data structures represented by these tags
 * should each have size `number_of_radial_points *
 * Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)`. Providing
 * buffers associated with these tags is necessary for the use of the aggregated
 * computation `mutate_all_swsh_derivatives_for_tag`.
 */
using all_transform_buffer_tags =
    tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
        all_swsh_derivative_tags,
        tmpl::bind<Spectral::Swsh::coefficient_buffer_tags_for_derivative_tag,
                   tmpl::_1>>>>;

namespace detail {
template <typename Tag>
struct additional_pre_swsh_derivative_tags_for {
  using type = tmpl::conditional_t<cpp17::is_same_v<Tag, Tags::BondiH>,
                                   tmpl::list<Tags::Dy<Tag>>,
                                   tmpl::list<Tag, Tags::Dy<Tag>>>;
};
}  // namespace detail

/// Typelist of steps for `PreSwshDerivatives` mutations needed for scri+
/// computations
using all_pre_swsh_derivative_tags_for_scri =
    tmpl::list<Tags::Dy<Tags::Du<Tags::BondiJ>>,
               Tags::Dy<Tags::Dy<Tags::BondiW>>,
               Tags::Dy<Tags::Dy<Tags::Dy<Tags::BondiJ>>>,
               Tags::ComplexInertialRetardedTime>;

/*!
 * \brief A typelist for the full set of tags needed as direct or indirect
 * input to any `ComputeBondiIntegrand` that are computed any specialization of
 * `ComputePreSwshDerivatives`.
 *
 * \details This is provided for easy and maintainable construction of a
 * `Variables` or \ref DataBoxGroup with all of the quantities needed for a CCE
 * computation or component. The data structures represented by these tags
 * should each have size `number_of_radial_points *
 * Spectral::Swsh::number_of_swsh_collocation_points(l_max)`.
 */
using all_pre_swsh_derivative_tags =
    tmpl::remove_duplicates<tmpl::flatten<tmpl::list<
        tmpl::transform<
            bondi_hypersurface_step_tags,
            tmpl::bind<
                tmpl::list, pre_swsh_derivative_tags_to_compute_for<tmpl::_1>,
                detail::additional_pre_swsh_derivative_tags_for<tmpl::_1>>>,
        all_pre_swsh_derivative_tags_for_scri>>>;

/// Typelist of steps for `SwshDerivatives` mutations called on volume
/// quantities needed for scri+ computations
using all_swsh_derivative_tags_for_scri = tmpl::list<
    Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiU>,
                                     Spectral::Swsh::Tags::Eth>,
    Spectral::Swsh::Tags::Derivative<
        Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                         Spectral::Swsh::Tags::EthEthbar>,
        Spectral::Swsh::Tags::Ethbar>,
    Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::Du<Tags::BondiJ>>,
                                     Spectral::Swsh::Tags::Ethbar>,
    Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::Dy<Tags::BondiU>>,
                                     Spectral::Swsh::Tags::Ethbar>,
    Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiQ>,
                                     Spectral::Swsh::Tags::Ethbar>,
    Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiU>,
                                     Spectral::Swsh::Tags::Eth>,
    Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::Dy<Tags::BondiBeta>>,
                                     Spectral::Swsh::Tags::Eth>>;

/// Typelist of steps for `SwshDerivatives` mutations called on boundary
/// (angular grid only) quantities needed for scri+ computations
using all_boundary_swsh_derivative_tags_for_scri =
    tmpl::list<Spectral::Swsh::Tags::Derivative<
        Tags::ComplexInertialRetardedTime, Spectral::Swsh::Tags::EthEth>>;

}  // namespace Cce
