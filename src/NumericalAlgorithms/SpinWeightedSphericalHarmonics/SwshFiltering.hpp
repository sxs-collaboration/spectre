// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "Utilities/Gsl.hpp"

namespace Spectral {
namespace Swsh {
/// @{
/*!
 * \ingroup SwshGroup
 * \brief Filter a volume collocation set in the form of consecutive
 * libsharp-compatible spherical shells.
 *
 * \details Two separate filters are applied. First, an exponential radial
 * filter is applied to each radial ray, with parameters `exponential_alpha` and
 * `exponential_half_power` (see `Spectral::filtering::exponential_filter` for
 * details on these parameters). Next, a modal Heaviside angular filter is
 * applied which simply sets to zero all `l > filter_max_l` modes.
 * \note It is assumed that Gauss-Lobatto points are used for the radial
 * direction (as that is the representation for CCE evolution). If that is too
 * restrictive, this function will need generalization.
 * \warning In principle, the radial filter in this function could cache the
 * matrix used, but currently does not. If such a cache becomes desirable for
 * performance, care must be taken regarding the exponential parameters. An
 * implementation similar to `dg::Actions::ExponentialFilter` may be necessary.
 * \note  For comparisons with SpEC CCE, `exponential_half_power` of 8,
 * `exponential_alpha` of 108, and `filter_max_l` of `l_max - 3` should be used.
 * This gives a highly aggressive radial filter, though, and for runs not
 * attempting to compare with SpEC it is recommended to use smaller parameters
 * to preserve more of the radial modes.
 */
template <int Spin>
void filter_swsh_volume_quantity(
    gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> to_filter,
    size_t l_max, size_t filter_max_l, double exponential_alpha,
    size_t exponential_half_power, gsl::not_null<ComplexDataVector*> buffer,
    gsl::not_null<SpinWeighted<ComplexModalVector, Spin>*> transform_buffer);

template <int Spin>
void filter_swsh_volume_quantity(
    gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> to_filter,
    size_t l_max, size_t filter_max_l, double exponential_alpha,
    size_t exponential_half_power);
/// @}
/// @{
/*!
 * \ingroup SwshGroup
 * \brief Filter a volume collocation set in the form of consecutive
 * libsharp-compatible spherical shells.
 *
 * \details Two separate filters are applied. First, an exponential radial
 * filter is applied to each radial ray, with parameters `exponential_alpha` and
 * `exponential_half_power` (see `Spectral::filtering::exponential_filter` for
 * details on these parameters). Next, a modal Heaviside angular filter is
 * applied which simply sets to zero all `l > max_l` or `l < min_l` modes.
 * \note It is assumed that Gauss-Lobatto points are used for the radial
 * direction (as that is the representation for CCE evolution). If that is too
 * restrictive, this function will need generalization.
 * \warning In principle, the radial filter in this function could cache the
 * matrix used, but currently does not. If such a cache becomes desirable for
 * performance, care must be taken regarding the exponential parameters. An
 * implementation similar to `dg::Actions::ExponentialFilter` may be necessary.
 */
template <int Spin>
void filter_swsh_volume_quantity(
    gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> to_filter,
    size_t l_max, size_t filter_min_l, size_t filter_max_l,
    double exponential_alpha, size_t exponential_half_power,
    gsl::not_null<ComplexDataVector*> buffer,
    gsl::not_null<SpinWeighted<ComplexModalVector, Spin>*> transform_buffer);

template <int Spin>
void filter_swsh_volume_quantity(
    gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> to_filter,
    size_t l_max, size_t filter_min_l, size_t filter_max_l,
    double exponential_alpha, size_t exponential_half_power);
/// @}
/// @{
/*!
 * \ingroup SwshGroup
 * \brief Filter a libsharp-compatible set of collocation points on a spherical
 * surface.
 *
 * \details A modal Heaviside angular filter is applied which simply sets to
 * zero all `l > filter_max_l` modes.
 * \note For comparisons with SpEC CCE, `filter_max_l` of `l_max - 3` should be
 * used.
 */
template <int Spin>
void filter_swsh_boundary_quantity(
    gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> to_filter,
    size_t l_max, size_t filter_max_l,
    gsl::not_null<SpinWeighted<ComplexModalVector, Spin>*> transform_buffer);

template <int Spin>
void filter_swsh_boundary_quantity(
    gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> to_filter,
    size_t l_max, size_t filter_max_l);
/// @}
/// @{
/*!
 * \ingroup SwshGroup
 * \brief Filter a libsharp-compatible set of collocation points on a spherical
 * surface.
 *
 * \details A modal Heaviside angular filter is applied which simply sets to
 * zero all `l > max_l` and `l < min_l` modes.
 */
template <int Spin>
void filter_swsh_boundary_quantity(
    gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> to_filter,
    size_t l_max, size_t filter_min_l, size_t filter_max_l,
    gsl::not_null<SpinWeighted<ComplexModalVector, Spin>*> transform_buffer);

template <int Spin>
void filter_swsh_boundary_quantity(
    gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> to_filter,
    size_t l_max, size_t filter_min_l, size_t filter_max_l);
/// @}
}  // namespace Swsh
}  // namespace Spectral
