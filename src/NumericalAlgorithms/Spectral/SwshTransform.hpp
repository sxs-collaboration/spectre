// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>
#include <sharp_cxx.h>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "NumericalAlgorithms/Spectral/ComplexDataView.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Coefficients
// IWYU pragma: no_forward_declare SpinWeighted

class ComplexDataVector;
class ComplexModalVector;

namespace Spectral {
namespace Swsh {

namespace detail {
// libsharp has an internal maximum number of transforms that is not in the
// public interface, so we must hard-code its value here
static const size_t max_libsharp_transforms = 100;

// appends to a vector of pointers `collocation_data` the set of pointers
// associated with angular views of the provided collocation data `vector`. The
// associated views are appended into `collocation_views`. If `positive_spin` is
// false, this function will conjugate the views, and therefore potentially
// conjugate (depending on `Representation`) the input data `vectors`. This
// behavior is chosen to avoid frequent copies of the input data `vector` for
// optimization. The conjugation must be undone after the call to libsharp
// transforms using `conjugate_views` to restore the input data to its original
// state.
template <ComplexRepresentation Representation>
void append_libsharp_collocation_pointers(
    gsl::not_null<std::vector<double*>*> collocation_data,
    gsl::not_null<std::vector<ComplexDataView<Representation>>*>
        collocation_views,
    gsl::not_null<ComplexDataVector*> vector, size_t l_max,
    bool positive_spin) noexcept;

// Perform a conjugation on a vector of `ComplexDataView`s if `Spin` < 0. This
// is used for undoing the conjugation described in the code comment for
// `append_libsharp_collocation_pointers`
template <int Spin, ComplexRepresentation Representation>
SPECTRE_ALWAYS_INLINE void conjugate_views(
    gsl::not_null<std::vector<ComplexDataView<Representation>>*>
        collocation_views) noexcept {
  if (Spin < 0) {
    for (auto& view : *collocation_views) {
      view.conjugate();
    }
  }
}

// appends to a vector of pointers the set of pointers associated with
// angular views of the provided coefficient vector. When working with the
// libsharp coefficient representation, note the intricacies mentioned in
// the documentation for `TransformJob`.
void append_libsharp_coefficient_pointers(
    gsl::not_null<std::vector<std::complex<double>*>*> coefficient_data,
    gsl::not_null<ComplexModalVector*> vector, size_t l_max) noexcept;

// perform the actual libsharp execution calls on an input and output set of
// pointers. This function handles the complication of a limited maximum number
// of simultaneous transforms, performing multiple execution calls on pointer
// blocks if necessary.
template <ComplexRepresentation Representation>
void execute_libsharp_transform_set(
    const sharp_jobtype& jobtype, int spin,
    gsl::not_null<std::vector<std::complex<double>*>*> coefficient_data,
    gsl::not_null<std::vector<double*>*> collocation_data,
    gsl::not_null<const CollocationMetadata<Representation>*>
        collocation_metadata,
    const sharp_alm_info* alm_info, size_t num_transforms) noexcept;

// template 'implementation' for the `swsh_transform` function below which
// performs an arbitrary number of transforms, and places them in the same
// number of destination modal containers passed by pointer.
// template notes: the parameter pack represents a collection of ordered modal
// data passed by pointer, followed by ordered nodal data passed by const
// reference. The first modal value is separated to infer the `Spin` template
// parameter.
template <ComplexRepresentation Representation, int Spin,
          typename... ModalThenNodalTypes, size_t... Is>
void swsh_transform_impl(
    size_t l_max, size_t number_of_radial_points,
    std::index_sequence<Is...> /*meta*/,
    gsl::not_null<SpinWeighted<ComplexModalVector, Spin>*> first_coefficient,
    const ModalThenNodalTypes&... coefficients_then_collocations) noexcept;

// template 'implementation' for the `inverse_swsh_transform` function below
// which performs an arbitrary number of transforms, and places them in the same
// number of destination nodal containers passed by pointer.
// template notes: the parameter pack represents a collection of ordered nodal
// data passed by pointer, followed by ordered modal data passed by const
// reference. The first nodal value is separated to infer the `Spin` template
// parameter.
template <ComplexRepresentation Representation, int Spin,
          typename... NodalThenModalTypes, size_t... Is>
void inverse_swsh_transform_impl(
    size_t l_max, size_t number_of_radial_points,
    std::index_sequence<Is...> /*meta*/,
    gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> first_collocation,
    const NodalThenModalTypes&... collocations_then_coefficients) noexcept;

// forward declaration for friend specification of helper from
// `SwshDerivatives.hpp`
template <typename Transform, typename FullTagList>
struct dispatch_to_transform;
}  // namespace detail

/*!
 * \ingroup SwshGroup
 * \brief Perform a forward libsharp spin-weighted spherical harmonic transform
 * on any number of supplied `SpinWeighted<ComplexDataVector, Spin>`.
 *
 * \details This function places the result in one or more
 * `SpinWeighted<ComplexModalVector, Spin>` returned via provided
 * pointer.  This is a simpler interface to the same functionality as the
 * \ref DataBoxGroup mutation compatible `SwshTransform`.
 *
 * The following parameter ordering for the multiple input interface is enforced
 * by the interior function calls, but is not obvious from the explicit
 * parameter packs in this function signature:
 *
 * - `size_t l_max` : angular resolution for the transform
 * - `size_t number_of_radial_points` : radial resolution (number of consecutive
 * transforms in each of the vectors)
 * - any number of `gsl::not_null<SpinWeighted<ComplexModalVector,
 * Spin>*>`, all sharing the same `Spin` : The return-by-pointer containers for
 * the transformed modal quantities
 * - the same number of `const SpinWeighted<ComplexDataVector, Spin>&`, with the
 * same `Spin` as the previous function argument set : The input containers of
 * nodal spin-weighted spherical harmonic data.
 *
 * template parameters:
 * - `Representation`: Either `ComplexRepresentation::Interleaved` or
 * `ComplexRepresentation::RealsThenImags`, indicating the representation for
 * intermediate steps of the transformation. The two representations will give
 * identical results but may help or hurt performance depending on the task.
 * If unspecified, defaults to `ComplexRepresentation::Interleaved`.
 *
 * The result is a set of libsharp-compatible coefficients.
 * \see SwshTransform for more details on the mathematics of the libsharp
 * data structures.
 *
 * \warning The collocation data is taken by const reference, but can be
 * temporarily altered in-place during intermediate parts of the computation.
 * The input data is guaranteed to return to its original state by the end of
 * the function. In a setting in which multiple threads access the same data
 * passed as input to this function, a lock must be used to prevent access
 * during the execution of the transform.
 */
template <
    ComplexRepresentation Representation = ComplexRepresentation::Interleaved,
    int Spin, typename... ModalThenNodalTypes>
void swsh_transform(
    const size_t l_max, const size_t number_of_radial_points,
    const gsl::not_null<SpinWeighted<ComplexModalVector, Spin>*>
        first_coefficient,
    const ModalThenNodalTypes&... coefficients_then_collocations) noexcept {
  static_assert(
      sizeof...(ModalThenNodalTypes) % 2 == 1,
      "The number of modal arguments passed to swsh_transform must equal the "
      "number of nodal arguments, so the total number of arguments must be "
      "even.");
  detail::swsh_transform_impl<Representation>(
      l_max, number_of_radial_points,
      std::make_index_sequence<(sizeof...(coefficients_then_collocations) + 1) /
                               2>{},
      first_coefficient, coefficients_then_collocations...);
}

/*!
 * \ingroup SwshGroup
 * \brief Perform a forward libsharp spin-weighted spherical harmonic transform
 * on a single supplied `SpinWeighted<ComplexDataVector, Spin>`.
 *
 * \details This function returns a `SpinWeighted<ComplexModalVector, Spin>` by
 * value (causing an allocation). This is a simpler interface to the same
 * functionality as the \ref DataBoxGroup mutation compatible `SwshTransform`.
 * If you have two or more vectors to transform at once, consider the
 * pass-by-pointer version of `Spectral::Swsh::swsh_transform` or the \ref
 * DataBoxGroup interface `InverseSwshTransform`, as they are more efficient for
 * performing several transforms at once.
 *
 * The result is a set of libsharp-compatible coefficients.
 * \see SwshTransform for more details on the mathematics of the libsharp
 * data structures.
 *
 * \warning The collocation data is taken by const reference, but can be
 * temporarily altered in-place during intermediate parts of the computation.
 * The input data is guaranteed to return to its original state by the end of
 * the function. In a setting in which multiple threads access the same data
 * passed as input to this function, a lock must be used to prevent access
 * during the execution of the transform.
 */
template <
    ComplexRepresentation Representation = ComplexRepresentation::Interleaved,
    int Spin>
SpinWeighted<ComplexModalVector, Spin> swsh_transform(
    size_t l_max, size_t number_of_radial_points,
    const SpinWeighted<ComplexDataVector, Spin>& collocation) noexcept;

/*!
 * \ingroup SwshGroup
 * \brief Perform an inverse libsharp spin-weighted spherical harmonic transform
 * on any number of supplied `SpinWeighted<ComplexModalVector, Spin>`.
 *
 * \details This function places the result in one or more
 * `SpinWeighted<ComplexDataVector, Spin>` returned via provided
 * pointer.  This is a simpler interface to the same functionality as the
 * \ref DataBoxGroup mutation compatible `InverseSwshTransform`.
 *
 * The following parameter ordering for the multiple input interface is enforced
 * by the interior function calls, but is not obvious from the explicit
 * parameter packs in this function signature:
 *
 * - `size_t l_max` : angular resolution for the transform
 * - `size_t number_of_radial_points` : radial resolution (number of consecutive
 * transforms in each of the vectors)
 * - any number of `gsl::not_null<SpinWeighted<ComplexDataVector,
 * Spin>*>`, all sharing the same `Spin` : The return-by-pointer containers for
 * the transformed nodal quantities
 * - the same number of `const SpinWeighted<ComplexModalVector, Spin>&`, with
 * the same `Spin` as the previous function argument set : The input containers
 * of modal spin-weighted spherical harmonic data.
 *
 * template parameters:
 * - `Representation`: Either `ComplexRepresentation::Interleaved` or
 * `ComplexRepresentation::RealsThenImags`, indicating the representation for
 * intermediate steps of the transformation. The two representations will give
 * identical results but may help or hurt performance depending on the task.
 * If unspecified, defaults to `ComplexRepresentation::Interleaved`.
 *
 * The input is a set of libsharp-compatible coefficients.
 * \see SwshTransform for more details on the mathematics of the libsharp
 * data structures.
 */
template <
    ComplexRepresentation Representation = ComplexRepresentation::Interleaved,
    int Spin, typename... NodalThenModalTypes>
void inverse_swsh_transform(
    const size_t l_max, const size_t number_of_radial_points,
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*>
        first_collocation,
    const NodalThenModalTypes&... collocations_then_coefficients) noexcept {
  static_assert(
      sizeof...(NodalThenModalTypes) % 2 == 1,
      "The number of nodal arguments passed to inverse_swsh_transform must "
      "equal the number of modal arguments, so the total number of arguments "
      "must be even.");
  detail::inverse_swsh_transform_impl<Representation>(
      l_max, number_of_radial_points,
      std::make_index_sequence<(sizeof...(collocations_then_coefficients) + 1) /
                               2>{},
      first_collocation, collocations_then_coefficients...);
}

/*!
 * \ingroup SwshGroup
 * \brief Perform an inverse libsharp spin-weighted spherical harmonic transform
 * on a single supplied `SpinWeighted<ComplexModalVector, Spin>`.
 *
 * \details This function returns a `SpinWeighted<ComplexDataVector, Spin>` by
 * value (causing an allocation). This is a simpler interface to the same
 * functionality as the \ref DataBoxGroup mutation compatible
 * `InverseSwshTransform`. If you have two or more vectors to transform at once,
 * consider the pass-by-pointer version of
 * `Spectral::Swsh::inverse_swsh_transform` or the \ref DataBoxGroup interface
 * `SwshTransform`, as they are more efficient for performing several transforms
 * at once.
 *
 * The input is a set of libsharp-compatible coefficients.
 * \see SwshTransform for more details on the mathematics of the libsharp
 * data structures.
 */
template <
    ComplexRepresentation Representation = ComplexRepresentation::Interleaved,
    int Spin>
SpinWeighted<ComplexDataVector, Spin> inverse_swsh_transform(
    size_t l_max, size_t number_of_radial_points,
    const SpinWeighted<ComplexModalVector, Spin>&
        libsharp_coefficients) noexcept;

/*!
 * \ingroup SwshGroup
 * \brief A \ref DataBoxGroup mutate-compatible computational struct for
 * performing several spin-weighted spherical harmonic transforms. Internally
 * dispatches to libsharp.
 *
 * \details
 * Template Parameters:
 * - `Representation`: The element of the `ComplexRepresentation` enum which
 *   parameterizes how the data passed to libsharp will be represented. Two
 *   options are available:
 *  - `ComplexRepresentation:Interleaved`: indicates that the real and imaginary
 *    parts of the collocation values will be passed to libsharp as pointers
 *    into existing complex data structures, minimizing copies, but maintaining
 *    a stride of 2 for 'adjacent' real or imaginary values.
 *  - `ComplexRepresentation::RealsThenImags`: indicates that the real and
 *    imaginary parts of the collocation values will be passed to libsharp as
 *    separate contiguous blocks. At current, this introduces both allocations
 *    and copies.  **optimization note** In the future most of the allocations
 *    can be aggregated to calling code, which would pass in buffers by
 *    `not_null` pointers.
 *
 *  For performance-sensitive code, both options should be tested, as each
 *  strategy has trade-offs.
 * - `TagList`: A `tmpl::list` of Tags to be forward transformed. The tags must
 * represent the nodal data.
 *
 * \note The signs obtained from libsharp transformations must be handled
 * carefully. (In particular, it does not use the sign convention you will find
 * in [Wikipedia]
 * (https://en.wikipedia.org/wiki/Spin-weighted_spherical_harmonics)).
 * - libsharp deals only with the transformation of real values, so
 *   transformation of complex values must be done for real and imaginary parts
 *   separately.
 * - due to only transforming real components, it stores only the positive
 *   \f$m\f$ modes (as the rest would be redundant). Therefore, the negative
 *   \f$m\f$ modes must be inferred from conjugation and further sign changes.
 * - libsharp only has the capability of transforming positive spin weighted
 *   quantities. Therefore, additional steps are taken (involving further
 *   conjugation of the provided data) in order to use those utilities on
 *   negative spin-weighted quantities. The resulting modes have yet more sign
 *   changes that must be taken into account.
 *
 * The decomposition resulting from the libsharp transform for spin \f$s\f$ and
 * complex spin-weighted \f${}_s f\f$ can be represented mathematically as:
 *
 * \f{align*}{
 * {}_s f(\theta, \phi) = \sum_{\ell = 0}^{\ell_\mathrm{max}} \Bigg\{&
 * \left(\sum_{m = 0}^{\ell} a^\mathrm{sharp, real}_{l m} {}_s Y_{\ell
 * m}^\mathrm{sharp, real}\right) + \left(\sum_{m=1}^{\ell}
 * \left(a^\mathrm{sharp, real}_{l m}{}\right)^*
 * {}_s Y_{\ell -m}^\mathrm{sharp, real}\right) \notag\\
 * &+ i \left[\left(\sum_{m = 0}^{\ell} a^\mathrm{sharp, imag}_{l m} {}_s
 * Y_{\ell m}^\mathrm{sharp, imag}\right) + \left(\sum_{m=1}^{\ell}
 * \left(a^\mathrm{sharp, imag}_{l m}{}\right)^* {}_s Y_{\ell -m}^\mathrm{sharp,
 * imag} \right)\right] \Bigg\},
 * \f}
 *
 * where
 *
 * \f{align*}{
 * {}_s Y_{\ell m}^\mathrm{sharp, real} &=
 * \begin{cases}
 * (-1)^{s + 1} {}_s Y_{\ell m}, & \mathrm{for}\; s < 0, m \ge 0 \\
 * {}_s Y_{\ell m}, & \mathrm{for}\; s = 0, m \ge 0 \\
 * - {}_s Y_{\ell m}, & \mathrm{for}\; s > 0, m \ge 0 \\
 * (-1)^{s + m + 1} {}_s Y_{\ell m}, & \mathrm{for}\; s < 0, m < 0 \\
 * (-1)^{m} {}_s Y_{\ell m}, & \mathrm{for}\; s = 0, m < 0 \\
 * (-1)^{m + 1} {}_s Y_{\ell m}, & \mathrm{for}\; s > 0, m < 0
 * \end{cases} \\
 * &\equiv {}_s S_{\ell m}^{\mathrm{real}} {}_s Y_{\ell m}\\
 * {}_s Y_{\ell m}^\mathrm{sharp, imag} &=
 * \begin{cases}
 * (-1)^{s + 1} {}_s Y_{\ell m}, & \mathrm{for}\; s < 0, m \ge 0 \\
 * -{}_s Y_{\ell m}, & \mathrm{for}\; s = 0, m \ge 0 \\
 * {}_s Y_{\ell m}, & \mathrm{for}\; s > 0, m \ge 0 \\
 * (-1)^{s + m} {}_s Y_{\ell m}, & \mathrm{for}\; s < 0, m < 0 \\
 * (-1)^{m} {}_s Y_{\ell m}, & \mathrm{for}\; s = 0, m < 0 \\
 * (-1)^{m + 1} {}_s Y_{\ell m}, & \mathrm{for}\; s > 0, m < 0
 * \end{cases} \\
 * &\equiv {}_s S_{\ell m}^{\mathrm{real}} {}_s Y_{\ell m},
 * \f}
 *
 * where the unadorned \f${}_s Y_{\ell m}\f$ on the right-hand-sides follow the
 * (older) convention of \cite Goldberg1966uu. Note the phase in these
 * expressions is not completely standardized, so should be checked carefully
 * whenever coefficient data is directly manipulated.
 *
 * For reference, we can compute the relation between Goldberg spin-weighted
 * moments \f${}_s f_{\ell m}\f$, defined as
 *
 * \f[ {}_s f(\theta, \phi) = \sum_{\ell = 0}^{\ell_\mathrm{max}} \sum_{m =
 * -\ell}^{\ell} {}_s f_{\ell m} {}_s Y_{\ell m}
 * \f]
 *
 * so,
 * \f[
 * {}_s f_{\ell m} =
 * \begin{cases}
 * a_{\ell m}^{\mathrm{sharp}, \mathrm{real}}  {}_s S_{\ell m}^{\mathrm{real}} +
 * a_{\ell m}^{\mathrm{sharp}, \mathrm{imag}}  {}_s S_{\ell m}^{\mathrm{imag}} &
 * \mathrm{for} \; m \ge 0 \\
 * \left(a_{\ell -m}^{\mathrm{sharp}, \mathrm{real}}\right)^* {}_s S_{\ell
 * m}^{\mathrm{real}} + \left(a_{\ell -m}^{\mathrm{sharp},
 * \mathrm{imag}}\right)^* {}_s S_{\ell m}^{\mathrm{imag}} &
 * \mathrm{for} \; m < 0 \\
 * \end{cases} \f]
 *
 *
 * The resulting coefficients \f$a_{\ell m}\f$ are stored in a triangular,
 * \f$\ell\f$-varies-fastest configuration. So, for instance, the first
 * \f$\ell_\mathrm{max}\f$ entries contain the coefficients for \f$m=0\f$ and
 * all \f$\ell\f$s, and the next \f$\ell_\mathrm{max} - 1\f$ entries contain the
 * coefficients for \f$m=1\f$ and \f$1 \le \ell \le \ell_\mathrm{max} \f$, and
 * so on.
 */
template <typename TagList, ComplexRepresentation Representation =
                                ComplexRepresentation::Interleaved>
struct SwshTransform;

/// \cond
template <typename... TransformTags, ComplexRepresentation Representation>
struct SwshTransform<tmpl::list<TransformTags...>, Representation> {
  static constexpr int spin =
      tmpl::front<tmpl::list<TransformTags...>>::type::type::spin;

  static_assert(tmpl2::flat_all_v<TransformTags::type::type::spin == spin...>,
                "All Tags in TagList submitted to SwshTransform must have the "
                "same spin weight.");

  using return_tags = tmpl::list<Tags::SwshTransform<TransformTags>...>;
  using argument_tags = tmpl::list<TransformTags..., Tags::LMaxBase,
                                   Tags::NumberOfRadialPointsBase>;

  static void apply(
      const gsl::not_null<
          typename Tags::SwshTransform<TransformTags>::type*>... coefficients,
      const typename TransformTags::type&... collocations, const size_t l_max,
      const size_t number_of_radial_points) noexcept {
    // forward to the version which takes parameter packs of vectors
    apply_to_vectors(make_not_null(&get(*coefficients))...,
                     get(collocations)..., l_max, number_of_radial_points);
  }

  // the convenience transform functions call the private member directly, so
  // need to be friends
  // redundant declaration due to forward declaration needs in earlier part of
  // file and friend requirements
  template <ComplexRepresentation FriendRepresentation, int FriendSpin,
            typename... CoefficientThenCollocationTypes, size_t... Is>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend void detail::swsh_transform_impl(
      size_t, size_t, std::index_sequence<Is...>,
      gsl::not_null<SpinWeighted<ComplexModalVector, FriendSpin>*>,
      const CoefficientThenCollocationTypes&...) noexcept;

  // friend declaration for helper function from `SwshDerivatives.hpp`
  template <typename Transform, typename FullTagList>
  friend struct detail::dispatch_to_transform;

 private:
  static void apply_to_vectors(
      const gsl::not_null<typename Tags::SwshTransform<
          TransformTags>::type::type*>... coefficients,
      const typename TransformTags::type::type&... collocations, size_t l_max,
      size_t number_of_radial_points) noexcept;
};
/// \endcond

/*!
 * \ingroup SwshGroup
 * \brief A \ref DataBoxGroup mutate-compatible computational struct for
 * performing several spin-weighted inverse spherical harmonic transforms.
 * Internally dispatches to libsharp.
 *
 * \details
 * Template Parameters:
 * - `Representation`: The element of the `ComplexRepresentation` enum which
 *   parameterizes how the data passed to libsharp will be represented. Two
 *   options are available:
 *  - `ComplexRepresentation:Interleaved`: indicates that the real and imaginary
 *    parts of the collocation values will be passed to libsharp as pointers
 *    into existing complex data structures, minimizing copies, but maintaining
 *    a stride of 2 for 'adjacent' real or imaginary values.
 *  - `ComplexRepresentation::RealsThenImags`: indicates that the real and
 *    imaginary parts of the collocation values will be passed to libsharp as
 *    separate contiguous blocks. At current, this introduces both allocations
 *    and copies.  **optimization note** In the future most of the allocations
 *    can be aggregated to calling code, which would pass in buffers by
 *    `not_null` pointers.
 *
 *  For performance-sensitive code, both options should be tested, as each
 *  strategy has trade-offs.
 * - `TagList`: A `tmpl::list` of Tags to be inverse transformed. The tags must
 * represent the nodal data being transformed to.
 *
 * \see `SwshTransform` for mathematical notes regarding the libsharp modal
 * representation taken as input for this computational struct.
 */
template <typename TagList, ComplexRepresentation Representation =
                                ComplexRepresentation::Interleaved>
struct InverseSwshTransform;

/// \cond
template <typename... TransformTags, ComplexRepresentation Representation>
struct InverseSwshTransform<tmpl::list<TransformTags...>, Representation> {
  static constexpr int spin =
      tmpl::front<tmpl::list<TransformTags...>>::type::type::spin;

  static_assert(
      tmpl2::flat_all_v<TransformTags ::type::type::spin == spin...>,
      "All Tags in TagList submitted to InverseSwshTransform must have the "
      "same spin weight.");

  using return_tags = tmpl::list<TransformTags...>;
  using argument_tags =
      tmpl::list<Tags::SwshTransform<TransformTags>..., Tags::LMaxBase,
                 Tags::NumberOfRadialPointsBase>;

  static void apply(
      const gsl::not_null<typename TransformTags::type*>... collocations,
      const typename Tags::SwshTransform<TransformTags>::type&... coefficients,
      const size_t l_max, const size_t number_of_radial_points) noexcept {
    // forward to the version which takes parameter packs of vectors
    apply_to_vectors(make_not_null(&get(*collocations))...,
                     get(coefficients)..., l_max, number_of_radial_points);
  }

  // the convenience transform functions call the private member directly, so
  // need to be friends
  // redundant declaration due to forward declaration needs in earlier part of
  // file and friend requirements
  template <ComplexRepresentation FriendRepresentation, int FriendSpin,
            typename... CollocationThenCoefficientTypes, size_t... Is>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend void detail::inverse_swsh_transform_impl(
      size_t, size_t, std::index_sequence<Is...>,
      gsl::not_null<SpinWeighted<ComplexDataVector, FriendSpin>*>,
      const CollocationThenCoefficientTypes&...) noexcept;

  // friend declaration for helper function from `SwshDerivatives.hpp`
  template <typename Transform, typename FullTagList>
  friend struct detail::dispatch_to_transform;

 private:
  static void apply_to_vectors(
      const gsl::not_null<typename TransformTags::type::type*>... collocations,
      const typename Tags::SwshTransform<
          TransformTags>::type::type&... coefficients,
      size_t l_max, size_t number_of_radial_points) noexcept;
};

template <typename... TransformTags, ComplexRepresentation Representation>
void SwshTransform<tmpl::list<TransformTags...>, Representation>::
    apply_to_vectors(const gsl::not_null<typename Tags::SwshTransform<
                         TransformTags>::type::type*>... coefficients,
                     const typename TransformTags::type::type&... collocations,
                     const size_t l_max,
                     const size_t number_of_radial_points) noexcept {
  EXPAND_PACK_LEFT_TO_RIGHT(coefficients->destructive_resize(
      size_of_libsharp_coefficient_vector(l_max) * number_of_radial_points));

  // assemble a list of pointers into the collocation point data. This is
  // required because libsharp expects pointers to pointers.
  std::vector<detail::ComplexDataView<Representation>> pre_transform_views;
  pre_transform_views.reserve(number_of_radial_points *
                              sizeof...(TransformTags));
  std::vector<double*> pre_transform_collocation_data;
  pre_transform_collocation_data.reserve(2 * number_of_radial_points *
                                         sizeof...(TransformTags));

  // clang-tidy: const-cast, object is temporarily modified and returned to
  // original state
  EXPAND_PACK_LEFT_TO_RIGHT(detail::append_libsharp_collocation_pointers(
      make_not_null(&pre_transform_collocation_data),
      make_not_null(&pre_transform_views),
      make_not_null(&const_cast<typename TransformTags::type::type&>(  // NOLINT
                         collocations)
                         .data()),
      l_max, spin >= 0));

  std::vector<std::complex<double>*> post_transform_coefficient_data;
  post_transform_coefficient_data.reserve(2 * number_of_radial_points *
                                          sizeof...(TransformTags));

  EXPAND_PACK_LEFT_TO_RIGHT(detail::append_libsharp_coefficient_pointers(
      make_not_null(&post_transform_coefficient_data),
      make_not_null(&coefficients->data()), l_max));

  const size_t num_transforms =
      (spin == 0 ? 2 : 1) * number_of_radial_points * sizeof...(TransformTags);
  const auto* collocation_metadata =
      &cached_collocation_metadata<Representation>(l_max);
  const auto* alm_info =
      cached_coefficients_metadata(l_max).get_sharp_alm_info();

  detail::execute_libsharp_transform_set(
      SHARP_MAP2ALM, spin, make_not_null(&post_transform_coefficient_data),
      make_not_null(&pre_transform_collocation_data),
      make_not_null(collocation_metadata), alm_info, num_transforms);

  detail::conjugate_views<spin>(make_not_null(&pre_transform_views));
}

template <typename... TransformTags, ComplexRepresentation Representation>
void InverseSwshTransform<tmpl::list<TransformTags...>, Representation>::
    apply_to_vectors(const gsl::not_null<
                         typename TransformTags::type::type*>... collocations,
                     const typename Tags::SwshTransform<
                         TransformTags>::type::type&... coefficients,
                     const size_t l_max,
                     const size_t number_of_radial_points) noexcept {
  EXPAND_PACK_LEFT_TO_RIGHT(collocations->destructive_resize(
      number_of_swsh_collocation_points(l_max) * number_of_radial_points));

  std::vector<std::complex<double>*> pre_transform_coefficient_data;
  pre_transform_coefficient_data.reserve(2 * number_of_radial_points *
                                         sizeof...(TransformTags));
  // clang-tidy: const-cast, object is temporarily modified and returned to
  // original state
  EXPAND_PACK_LEFT_TO_RIGHT(detail::append_libsharp_coefficient_pointers(
      make_not_null(&pre_transform_coefficient_data),
      make_not_null(&const_cast<typename Tags::SwshTransform<  // NOLINT
                         TransformTags>::type::type&>(coefficients)
                         .data()),
      l_max));

  std::vector<detail::ComplexDataView<Representation>> post_transform_views;
  post_transform_views.reserve(number_of_radial_points *
                               sizeof...(TransformTags));
  std::vector<double*> post_transform_collocation_data;
  post_transform_collocation_data.reserve(2 * number_of_radial_points *
                                          sizeof...(TransformTags));

  EXPAND_PACK_LEFT_TO_RIGHT(detail::append_libsharp_collocation_pointers(
      make_not_null(&post_transform_collocation_data),
      make_not_null(&post_transform_views),
      make_not_null(&collocations->data()), l_max, true));

  // libsharp considers two arrays per transform when spin is not zero.
  const size_t num_transforms =
      (spin == 0 ? 2 : 1) * number_of_radial_points * sizeof...(TransformTags);
  const auto* collocation_metadata =
      &cached_collocation_metadata<Representation>(l_max);
  const auto* alm_info =
      cached_coefficients_metadata(l_max).get_sharp_alm_info();

  detail::execute_libsharp_transform_set(
      SHARP_ALM2MAP, spin, make_not_null(&pre_transform_coefficient_data),
      make_not_null(&post_transform_collocation_data),
      make_not_null(collocation_metadata), alm_info, num_transforms);

  detail::conjugate_views<spin>(make_not_null(&post_transform_views));

  // The inverse transformed collocation data has just been placed in the
  // memory blocks controlled by the `ComplexDataView`s. Finally, that data
  // must be flushed back to the Variables.
  for (auto& view : post_transform_views) {
    view.copy_back_to_source();
  }
}
/// \endcond

namespace detail {

template <ComplexRepresentation Representation, int Spin,
          typename... ModalThenNodalTypes, size_t... Is>
void swsh_transform_impl(
    const size_t l_max, const size_t number_of_radial_points,
    std::index_sequence<Is...> /*meta*/,
    const gsl::not_null<SpinWeighted<ComplexModalVector, Spin>*>
        first_coefficient,
    const ModalThenNodalTypes&... coefficients_then_collocations) noexcept {
  SwshTransform<
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<Is, ComplexDataVector>,
                                      std::integral_constant<int, Spin>>...>>::
      apply_to_vectors(first_coefficient, coefficients_then_collocations...,
                       l_max, number_of_radial_points);
}

template <ComplexRepresentation Representation, int Spin,
          typename... NodalThenModalTypes, size_t... Is>
void inverse_swsh_transform_impl(
    const size_t l_max, const size_t number_of_radial_points,
    std::index_sequence<Is...> /*meta*/,
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*>
        first_collocation,
    const NodalThenModalTypes&... collocations_then_coefficients) noexcept {
  InverseSwshTransform<
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<Is, ComplexDataVector>,
                                      std::integral_constant<int, Spin>>...>>::
      apply_to_vectors(first_collocation, collocations_then_coefficients...,
                       l_max, number_of_radial_points);
}

// A metafunction for binning a provided tag list into `SwshTransform` objects
// according to spin-weight
template <int MinSpin, ComplexRepresentation Representation, typename TagList,
          typename IndexSequence>
struct make_transform_list_impl;

template <int MinSpin, ComplexRepresentation Representation, typename TagList,
          int... Is>
struct make_transform_list_impl<MinSpin, Representation, TagList,
                                std::integer_sequence<int, Is...>> {
  using type = tmpl::flatten<tmpl::list<tmpl::conditional_t<
      not std::is_same_v<get_tags_with_spin<Is + MinSpin, TagList>,
                         tmpl::list<>>,
      SwshTransform<get_tags_with_spin<Is + MinSpin, TagList>, Representation>,
      tmpl::list<>>...>>;
};

// A metafunction for binning a provided tag list into `InverseSwshTransform`
// objects according to spin-weight
template <int MinSpin, ComplexRepresentation Representation, typename TagList,
          typename IndexSequence>
struct make_inverse_transform_list_impl;

template <int MinSpin, ComplexRepresentation Representation, typename TagList,
          int... Is>
struct make_inverse_transform_list_impl<MinSpin, Representation, TagList,
                                        std::integer_sequence<int, Is...>> {
  using type = tmpl::flatten<tmpl::list<tmpl::conditional_t<
      not std::is_same_v<get_tags_with_spin<Is + MinSpin, TagList>,
                         tmpl::list<>>,
      InverseSwshTransform<get_tags_with_spin<Is + MinSpin, TagList>,
                           Representation>,
      tmpl::list<>>...>>;
};
}  // namespace detail

// @{
/// \ingroup SwshGroup
/// \brief Assemble a `tmpl::list` of `SwshTransform`s or
/// `InverseSwshTransform`s given a list of tags `TagList` that need to be
/// transformed. The `Representation` is the
/// `Spectral::Swsh::ComplexRepresentation` to use for the transformations.
///
/// \details Up to five `SwshTransform`s or `InverseSwshTransform`s will be
/// returned, corresponding to the possible spin values. Any number of
/// transformations are aggregated into that set of `SwshTransform`s (or
/// `InverseSwshTransform`s). The number of transforms is up to five because the
/// libsharp utility only has capability to perform spin-weighted spherical
/// harmonic transforms for integer spin-weights from -2 to 2.
///
/// \snippet Test_SwshTransform.cpp make_transform_list
template <ComplexRepresentation Representation, typename TagList>
using make_transform_list = typename detail::make_transform_list_impl<
    -2, Representation, TagList,
    decltype(std::make_integer_sequence<int, 5>{})>::type;

template <ComplexRepresentation Representation, typename TagList>
using make_inverse_transform_list =
    typename detail::make_inverse_transform_list_impl<
        -2, Representation, TagList,
        decltype(std::make_integer_sequence<int, 5>{})>::type;
// @}

/// \ingroup SwshGroup
/// \brief Assemble a `tmpl::list` of `SwshTransform`s given a list of
/// `Spectral::Swsh::Tags::Derivative<Tag, Derivative>` that need to be
/// computed. The `SwshTransform`s constructed by this type alias correspond to
/// the `Tag`s in the list.
///
/// \details This is intended as a convenience utility for the first step of a
/// derivative routine, where one transforms the set of fields for which
/// derivatives are required.
///
/// \snippet Test_SwshTransform.cpp make_transform_from_derivative_tags
template <ComplexRepresentation Representation, typename DerivativeTagList>
using make_transform_list_from_derivative_tags =
    typename detail::make_transform_list_impl<
        -2, Representation,
        tmpl::transform<DerivativeTagList,
                        tmpl::bind<db::remove_tag_prefix, tmpl::_1>>,
        decltype(std::make_integer_sequence<int, 5>{})>::type;

/// \ingroup SwshGroup
/// \brief Convert spin-weighted spherical harmonic data to a new set of
/// collocation points (either downsampling or upsampling)
template <int Spin>
void interpolate_to_collocation(
    gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> target,
    const SpinWeighted<ComplexDataVector, Spin>& source, size_t target_l_max,
    size_t source_l_max, size_t number_of_radial_points) noexcept;

}  // namespace Swsh
}  // namespace Spectral
