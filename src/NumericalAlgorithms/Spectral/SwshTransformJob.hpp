// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <sharp_cxx.h>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/ComplexDataView.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Variables
// IWYU pragma: no_forward_declare Coefficients

namespace Spectral {
namespace Swsh {

/*!
 * \ingroup SwshGroup
 * \brief A class which gathers all necessary shared structure among several
 * spin-weighted spherical harmonic transforms and dispatches to libsharp. Each
 * `TransformJob` represents exactly one spin-weighted transform libsharp
 * execution call, and one inverse transform libsharp execution call.
 *
 *
 * \details
 * Template Parameters:
 * - `Spin`: The spin weight for all of the fields to transform with this job.
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
 * - `TagList`: A `tmpl::list` of Tags to be forward and/or backward transformed
 *   with the `TransformJob`.
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
template <int Spin, ComplexRepresentation Representation, typename TagList>
class TransformJob {
 public:
  static_assert(cpp17::is_same_v<get_tags_with_spin<Spin, TagList>, TagList>,
                "All Tags in TagList submitted to TransformJob must have the "
                "same spin weight as the TransformJob Spin template parameter");
  using CoefficientTagList = db::wrap_tags_in<Tags::SwshTransform, TagList>;
  static constexpr int spin = Spin;

  /// \brief Constructor for transform job. Both the `l_max` and
  /// `number_of_radial_grid_points` must be specified for the transformation to
  /// appropriately find all of the data in memory and associate it with the
  /// correct collocation grid.
  TransformJob(size_t l_max, size_t number_of_radial_grid_points) noexcept;

  /// \brief Helper function for determining the size of allocation necessary to
  /// store the coefficient data.
  ///
  /// \note This size is not the same as the collocation data size, as the
  /// coefficients are stored in the efficient 'triangular' form noted in the
  /// documentation for `TransformJob`.
  constexpr size_t coefficient_output_size() const noexcept {
    return number_of_swsh_coefficients(l_max_) * number_of_radial_grid_points_;
  }

  /// \brief Execute the forward spin-weighted spherical harmonic transform
  /// using libsharp.
  ///
  /// \param output A `Variables` which must contain `CoefficientTag<...>`s
  /// for each of the tags provided via `TagList`. The coefficients will be
  /// stored appropriately in this Variables.
  /// \param input A `Variables` which must contain each of the tags provided
  /// via `TagList`. The collocation points may be temporarily altered
  /// (conjugated) during the execution of the transform if the `Spin` is
  /// negative, but are reverted to their original state by the end of the
  /// function execution.
  template <typename InputVarsTagList, typename OutputVarsTagList>
  void execute_transform(
      gsl::not_null<Variables<OutputVarsTagList>*> output,
      gsl::not_null<Variables<InputVarsTagList>*> input) const noexcept;

  /// \brief Execute the inverse spin-weighted spherical harmonic transform
  /// using libsharp.
  ///
  /// \param input A `Variables` which must contain `CoefficientTag<...>`s
  /// for each of the tags provided via `TagList`. The coefficients may be
  /// temporarily altered (conjugated) during the execution of the transform if
  /// the `Spin` is negative, but are reverted to their original state by the
  /// end of the function execution.
  /// \param output A `Variables` which must contain each of the tags provided
  /// via `TagList`. The collocation data will be stored appropriately in this
  /// Variables.
  template <typename InputVarsTagList, typename OutputVarsTagList>
  void execute_inverse_transform(
      gsl::not_null<Variables<OutputVarsTagList>*> output,
      gsl::not_null<Variables<InputVarsTagList>*> input) const noexcept;

 private:
  size_t number_of_radial_grid_points_;
  size_t l_max_;
  sharp_alm_info* alm_info_;
  const Collocation<Representation>* collocation_metadata_;
};

template <int Spin, ComplexRepresentation Representation, typename TagList>
TransformJob<Spin, Representation, TagList>::TransformJob(
    const size_t l_max, const size_t number_of_radial_grid_points) noexcept
    : number_of_radial_grid_points_{number_of_radial_grid_points},
      l_max_{l_max},
      collocation_metadata_{&precomputed_collocation<Representation>(l_max_)} {
  alm_info_ = detail::precomputed_coefficients(l_max_).get_sharp_alm_info();
}

template <int Spin, ComplexRepresentation Representation, typename TagList>
template <typename InputVarsTagList, typename OutputVarsTagList>
void TransformJob<Spin, Representation, TagList>::execute_transform(
    const gsl::not_null<Variables<OutputVarsTagList>*> output,
    const gsl::not_null<Variables<InputVarsTagList>*> input) const noexcept {
  // assemble a list of pointers into the collocation point data. This is
  // required because libsharp expects pointers to pointers.
  std::vector<detail::ComplexDataView<Representation>> pre_transform_views;
  pre_transform_views.reserve(number_of_radial_grid_points_ *
                              tmpl::size<TagList>::value);
  std::vector<const double*> pre_transform_collocation_data;
  pre_transform_collocation_data.reserve(2 * number_of_radial_grid_points_ *
                                         tmpl::size<TagList>::value);

  // for each Tag and each slice block in the radial direction, construct a
  // ComplexDataView at the appropriate locations, then retrieve its real
  // and imaginary data and put it in subsequent slots in the
  // pre_transform_collocation_data.
  size_t num_transforms = 0;
  tmpl::for_each<TagList>([
    &pre_transform_collocation_data, &pre_transform_views, &num_transforms,
    &input, this
  ](auto x) noexcept {
    using transform_var_tag = typename decltype(x)::type;
    decltype(auto) input_vector = get(get<transform_var_tag>(*input)).data();
    for (size_t i = 0; i < number_of_radial_grid_points_; ++i) {
      pre_transform_views.push_back(detail::ComplexDataView<Representation>{
          make_not_null(&input_vector), collocation_metadata_->size(),
          i * collocation_metadata_->size()});
      pre_transform_collocation_data.push_back(
          pre_transform_views.back().real_data());
      // alteration needed because libsharp doesn't support negative spins
      if (spin < 0) {
        pre_transform_views.back().conjugate();
      }
      pre_transform_collocation_data.push_back(
          pre_transform_views.back().imag_data());
      // libsharp considers two arrays per transform when spin is not zero.
      num_transforms += (spin == 0 ? 2 : 1);
    }
  });

  std::vector<std::complex<double>*> post_transform_coefficient_data;
  post_transform_coefficient_data.reserve(2 * number_of_radial_grid_points_ *
                                          tmpl::size<TagList>::value);

  // assemble list of locations in the output Variables for the destinations
  // of the coefficient data
  tmpl::for_each<CoefficientTagList>([
    &post_transform_coefficient_data, &output, this
  ](auto x) noexcept {
    using coefficient_var_tag = typename decltype(x)::type;
    decltype(auto) output_vector =
        get(get<coefficient_var_tag>(*output)).data();
    for (size_t i = 0; i < number_of_radial_grid_points_; ++i) {
      // coefficients associated with the real part
      post_transform_coefficient_data.push_back(
          output_vector.data() + 2 * i * number_of_swsh_coefficients(l_max_));
      // coefficients associated with the imaginary part
      post_transform_coefficient_data.push_back(
          output_vector.data() +
          (2 * i + 1) * number_of_swsh_coefficients(l_max_));
    }
  });

  sharp_execute(SHARP_MAP2ALM, abs(spin),
                post_transform_coefficient_data.data(),
                pre_transform_collocation_data.data(),
                collocation_metadata_->get_sharp_geom_info(), alm_info_,
                num_transforms, SHARP_DP, nullptr, nullptr);

  // libsharp only has the ability to transform positive spins, so we have to
  // perform conjugation to deal with negative spins.
  if (spin < 0) {
    // fix the conjugation performed prior to transformation.
    for (auto& view : pre_transform_views) {
      view.conjugate();
    }
  }
}

template <int Spin, ComplexRepresentation Representation, typename TagList>
template <typename InputVarsTagList, typename OutputVarsTagList>
void TransformJob<Spin, Representation, TagList>::execute_inverse_transform(
    const gsl::not_null<Variables<OutputVarsTagList>*> output,
    const gsl::not_null<Variables<InputVarsTagList>*> input) const noexcept {
  std::vector<std::complex<double>*> pre_transform_coefficient_data;
  pre_transform_coefficient_data.reserve(2 * number_of_radial_grid_points_ *
                                         tmpl::size<CoefficientTagList>::value);
  // assemble list of locations in the input Variables for the locations
  // of the coefficient data
  tmpl::for_each<CoefficientTagList>([&pre_transform_coefficient_data, &input,
                                      this](auto x) {
    using coefficient_var_tag = typename decltype(x)::type;
    decltype(auto) input_vector = get(get<coefficient_var_tag>(*input)).data();
    for (size_t i = 0; i < number_of_radial_grid_points_; ++i) {
      pre_transform_coefficient_data.push_back(
          input_vector.data() + 2 * i * number_of_swsh_coefficients(l_max_));
      pre_transform_coefficient_data.push_back(
          input_vector.data() +
          (2 * i + 1) * number_of_swsh_coefficients(l_max_));
    }
  });

  std::vector<detail::ComplexDataView<Representation>> post_transform_views;
  post_transform_views.reserve(number_of_radial_grid_points_ *
                               tmpl::size<TagList>::value);
  std::vector<const double*> post_transform_collocation_data;
  post_transform_collocation_data.reserve(2 * number_of_radial_grid_points_ *
                                          tmpl::size<TagList>::value);

  // for each tag in `TagList`, construct a ComplexDataView at the
  // appropriate locations, then retrieve its real and imaginary data and put
  // it in subsequent slots in the pre_transform_collocation_data
  size_t num_transforms = 0;
  tmpl::for_each<TagList>([&post_transform_collocation_data, &num_transforms,
                           &post_transform_views, &output, this](auto x) {
    using transform_var_tag = typename decltype(x)::type;
    decltype(auto) output_vector = get(get<transform_var_tag>(*output)).data();
    for (size_t i = 0; i < number_of_radial_grid_points_; ++i) {
      post_transform_views.push_back(detail::ComplexDataView<Representation>{
          make_not_null(&output_vector), collocation_metadata_->size(),
          i * collocation_metadata_->size()});
      post_transform_collocation_data.push_back(
          post_transform_views.back().real_data());
      post_transform_collocation_data.push_back(
          post_transform_views.back().imag_data());
      // libsharp considers two arrays per transform when spin is not zero.
      num_transforms += (spin == 0 ? 2 : 1);
    }
  });

  sharp_execute(SHARP_ALM2MAP, abs(spin), pre_transform_coefficient_data.data(),
                post_transform_collocation_data.data(),
                collocation_metadata_->get_sharp_geom_info(), alm_info_,
                num_transforms, SHARP_DP, nullptr, nullptr);

  if (spin < 0) {
    // the conjugate is needed because libsharp doesn't handle negative spins.
    for (auto& view : post_transform_views) {
      view.conjugate();
    }
  }

  // The inverse transformed collocation data has just been placed in the
  // memory blocks controlled by the `ComplexDataView`s. Finally, that data
  // must be flushed back to the Variables.
  for (auto& view : post_transform_views) {
    view.copy_back_to_source();
  }
}

namespace detail {
template <typename Job>
struct transform_job_is_not_empty : std::true_type {};

template <int Spin, ComplexRepresentation Representation>
struct transform_job_is_not_empty<
    TransformJob<Spin, Representation, tmpl::list<>>> : std::false_type {};

template <int MinSpin, ComplexRepresentation Representation, typename TagList,
          typename IndexSequence>
struct make_swsh_transform_job_list_impl;

template <int MinSpin, ComplexRepresentation Representation, typename TagList,
          int... Is>
struct make_swsh_transform_job_list_impl<MinSpin, Representation, TagList,
                                         std::integer_sequence<int, Is...>> {
  using type = tmpl::filter<
      tmpl::list<TransformJob<Is + MinSpin, Representation,
                              get_tags_with_spin<Is + MinSpin, TagList>>...>,
      transform_job_is_not_empty<tmpl::_1>>;
};
}  // namespace detail

/// \ingroup SpectralGroup
/// \brief Assemble a `tmpl::list` of `TransformJob` given a list of tags
/// `TagList` that need to be transformed. The `Representation` is the
/// `ComplexRepresentation` to use for the transformations.
///
/// \details Up to five `TransformJob` will be returned, corresponding to
/// the possible spin values. Any number of transformations are aggregated into
/// that set of `TransformJob`s. The number of transforms is up to five because
/// the libsharp utility only has capability to perform spin-weighted spherical
/// harmonic transforms for integer spin-weights from -2 to 2.
///
/// \snippet Test_SwshTransformJob.cpp make_swsh_transform_job_list
template <ComplexRepresentation Representation, typename TagList>
using make_swsh_transform_job_list =
    typename detail::make_swsh_transform_job_list_impl<
        -2, Representation, TagList,
        decltype(std::make_integer_sequence<int, 5>{})>::type;

/// \ingroup SpectralGroup
/// \brief Assemble a `tmpl::list` of `TransformJob`s given a list of
/// `Derivative<Tag, Derivative>` that need to be computed. The
/// `TransformJob`s constructed by this type alias correspond to the
/// `Tag`s in the list.
///
/// \details This is intended as a convenience utility for the first step of a
/// derivative routine, where one transforms the set of fields for which
/// derivatives are required.
///
/// \snippet Test_SwshTransformJob.cpp make_swsh_transform_from_derivative_tags
template <ComplexRepresentation Representation, typename DerivativeTagList>
using make_swsh_transform_job_list_from_derivative_tags =
    typename detail::make_swsh_transform_job_list_impl<
        -2, Representation,
        tmpl::transform<DerivativeTagList,
                        tmpl::bind<db::remove_tag_prefix, tmpl::_1>>,
        decltype(std::make_integer_sequence<int, 5>{})>::type;
}  // namespace Swsh
}  // namespace Spectral
