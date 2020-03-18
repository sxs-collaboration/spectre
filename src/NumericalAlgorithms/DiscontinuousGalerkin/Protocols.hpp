// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"
#include "Utilities/TypeTraits/IsCallable.hpp"

namespace dg {
/// \ref protocols related to Discontinuous Galerkin functionality
namespace protocols {

namespace detail {
CREATE_HAS_TYPE_ALIAS(variables_tags)
CREATE_HAS_TYPE_ALIAS_V(variables_tags)
CREATE_HAS_TYPE_ALIAS(argument_tags)
CREATE_HAS_TYPE_ALIAS_V(argument_tags)
CREATE_HAS_TYPE_ALIAS(package_field_tags)
CREATE_HAS_TYPE_ALIAS_V(package_field_tags)
CREATE_HAS_TYPE_ALIAS(package_extra_tags)
CREATE_HAS_TYPE_ALIAS_V(package_extra_tags)
CREATE_IS_CALLABLE(package_data)

template <typename NumericalFluxType, typename ArgumentTags,
          typename PackageFieldTags, typename PackageExtraTags>
struct IsPackageDataCallableImpl;

template <typename NumericalFluxType, typename... ArgumentTags,
          typename... PackageFieldTags, typename... PackageExtraTags>
struct IsPackageDataCallableImpl<NumericalFluxType, tmpl::list<ArgumentTags...>,
                                 tmpl::list<PackageFieldTags...>,
                                 tmpl::list<PackageExtraTags...>>
    : is_package_data_callable_r_t<
          void, NumericalFluxType,
          gsl::not_null<db::item_type<PackageFieldTags>*>...,
          gsl::not_null<db::item_type<PackageExtraTags>*>...,
          db::const_item_type<ArgumentTags>...> {};

template <typename NumericalFluxType>
struct IsPackageDataCallable
    : IsPackageDataCallableImpl<
          NumericalFluxType, typename NumericalFluxType::argument_tags,
          typename NumericalFluxType::package_field_tags,
          typename NumericalFluxType::package_extra_tags> {};

template <typename NumericalFluxType, typename VariablesTags,
          typename PackageFieldTags, typename PackageExtraTags>
struct IsNumericalFluxCallableImpl;

template <typename NumericalFluxType, typename... VariablesTags,
          typename... PackageFieldTags, typename... PackageExtraTags>
struct IsNumericalFluxCallableImpl<
    NumericalFluxType, tmpl::list<VariablesTags...>,
    tmpl::list<PackageFieldTags...>, tmpl::list<PackageExtraTags...>>
    : tt::is_callable_t<NumericalFluxType,
                        gsl::not_null<db::item_type<VariablesTags>*>...,
                        db::const_item_type<PackageFieldTags>...,
                        db::const_item_type<PackageExtraTags>...,
                        db::const_item_type<PackageFieldTags>...,
                        db::const_item_type<PackageExtraTags>...> {};

template <typename NumericalFluxType>
struct IsNumericalFluxCallable
    : IsNumericalFluxCallableImpl<
          NumericalFluxType, typename NumericalFluxType::variables_tags,
          typename NumericalFluxType::package_field_tags,
          typename NumericalFluxType::package_extra_tags> {};
}  // namespace detail

/*!
 * \ingroup ProtocolsGroup
 * \brief Defines the interface for DG numerical fluxes
 *
 * This protocol defines the interface that a class must conform to so that it
 * can be used as a numerical flux in DG boundary schemes. Essentially, the
 * class must be able to compute the quantity \f$G\f$ that appears, for example,
 * in the strong first-order DG boundary scheme
 * \f$G_\alpha(n_i^\mathrm{int}, u_\alpha^\mathrm{int}, n_i^\mathrm{ext},
 * u_\alpha^\mathrm{ext}) - n_i^\mathrm{int} F^{i,\mathrm{int}}_\alpha\f$
 * where \f$u_\alpha\f$ are the system variables and \f$F^i_\alpha\f$ their
 * corresponding fluxes. See also Eq. (2.20) in \cite Teukolsky2015ega where the
 * quantity \f$G\f$ is denoted \f$n_i F^{i*}\f$, which is why we occasionally
 * refer to it as the "normal-dot-numerical-fluxes".
 *
 * Requires the `ConformingType` has these type aliases:
 * - `variables_tags`: A typelist of DataBox tags that the class computes
 * numerical fluxes for.
 * - `argument_tags`: A typelist of DataBox tags that will be retrieved on
 * interfaces and passed to the `package_data` function (see below). The
 * `ConformingType` may also have a `volume_tags` typelist that specifies the
 * subset of `argument_tags` that should be retrieved from the volume instead
 * of the interface.
 * - `package_field_tags`: A typelist of DataBox tags with `Tensor` types that
 * the `package_data` function will compute from the `argument_tags`. These
 * quantities will be made available on both sides of a mortar and passed to
 * the call operator to compute the numerical flux.
 * - `package_extra_tags`: Additional non-tensor tags that will be made
 * available on both sides of a mortar, e.g. geometric quantities.
 *
 * Requires the `ConformingType` has these member functions:
 * - `package_data`: Takes the types of the `package_field_tags` and the
 * `package_extra_tags` by `gsl::not_null` pointer, followed by the types of the
 * `argument_tags`.
 * - `operator()`: Takes the types of the `variables_tags` by `gsl::not_null`
 * pointer, followed by the types of the `package_field_tags` and the
 * `package_extra_tags` from the interior side of the mortar and from the
 * exterior side. Note that the data from the exterior side was computed
 * entirely with data from the neighboring element, including its interface
 * normal which is (at least when it's independent of the system variables)
 * opposite to the interior element's interface normal. Therefore, make sure to
 * take into account the sign flip for quantities that include the interface
 * normal.
 *
 * Here's an example for a simple "central" numerical flux
 * \f$G_\alpha(n_i^\mathrm{int}, u_\alpha^\mathrm{int}, n_i^\mathrm{ext},
 * u_\alpha^\mathrm{ext}) = \frac{1}{2}\left(n_i^\mathrm{int}
 * F^{i,\mathrm{int}}_\alpha - n_i^\mathrm{ext} F^{i,\mathrm{ext}}_\alpha
 * \right)\f$:
 *
 * \snippet DiscontinuousGalerkin/Test_Protocols.cpp numerical_flux_example
 *
 * Note that this numerical flux reduces to the interface average
 * \f$G_\alpha=\frac{n^\mathrm{int}_i}{2}\left(F^{i,\mathrm{int}}_\alpha +
 * F^{i,\mathrm{ext}}_\alpha\right)\f$ for the case where the interface normal
 * is independent of the system variables and therefore \f$n_i^\mathrm{ext} =
 * -n_i^\mathrm{int}\f$.
 */
template <typename ConformingType>
using NumericalFlux = std::conditional_t<
    tmpl2::flat_all_v<detail::has_variables_tags_v<ConformingType>,
                      detail::has_argument_tags_v<ConformingType>,
                      detail::has_package_field_tags_v<ConformingType>,
                      detail::has_package_extra_tags_v<ConformingType>>,
    cpp17::conjunction<detail::IsPackageDataCallable<ConformingType>,
                       detail::IsNumericalFluxCallable<ConformingType>>,
    std::false_type>;

}  // namespace protocols
}  // namespace dg
