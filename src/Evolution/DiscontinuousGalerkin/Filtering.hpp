// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Options.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace dg {
namespace Actions {
namespace ExponentialFilter_detail {
template <bool SameSize, bool SameList>
struct FilterAllEvolvedVars {
  template <typename EvolvedVarsTagList, typename... FilterTags>
  using f = std::integral_constant<bool, false>;
};

template <>
struct FilterAllEvolvedVars<true, false> {
  template <typename EvolvedVarsTagList, typename... FilterTags>
  using f =
      std::integral_constant<bool, tmpl2::flat_all_v<tmpl::list_contains_v<
                                       EvolvedVarsTagList, FilterTags>...>>;
};

template <>
struct FilterAllEvolvedVars<true, true> {
  template <typename EvolvedVarsTagList, typename... FilterTags>
  using f = std::integral_constant<bool, true>;
};
}  // namespace ExponentialFilter_detail

/// \cond
template <size_t FilterIndex, typename TagsToFilterList>
struct ExponentialFilter;
/// \endcond

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Applies an exponential filter to the specified tags.
 *
 * Applies an exponential filter in each logical direction to each component of
 * the tensors `TagsToFilter`. The exponential filter rescales the 1d modal
 * coefficients \f$c_i\f$ as:
 *
 * \f{align*}{
 *  c_i\to c_i \exp\left[-\alpha_{\mathrm{ef}}
 *   \left(\frac{i}{N}\right)^{2\beta_{\mathrm{ef}}}\right]
 * \f}
 *
 * where \f$N\f$ is the basis order, \f$\alpha_{\mathrm{ef}}\f$ determines how
 * much the coefficients are rescaled, and \f$\beta_{\mathrm{ef}}\f$ (given by
 * the `HalfPower` option) determines how aggressive/broad the filter is (lower
 * values means filtering more coefficients). Setting
 * \f$\alpha_{\mathrm{ef}}=36\f$ results in effectively zeroing the highest
 * coefficient (in practice it gets rescaled by machine epsilon). The same
 * \f$\alpha_{\mathrm{ef}}\f$ and \f$\beta_{\mathrm{ef}}\f$ are used in each
 * logical direction. For a discussion of filtering see section 5.3 of
 * \cite HesthavenWarburton.
 *
 * If different `Alpha` or `HalfPower` parameters are desired for different tags
 * then multiple `ExponentialFilter` actions must be inserted into the action
 * list with different `FilterIndex` values. In the input file these will be
 * specified as `ExpFilterFILTER_INDEX`, e.g. `ExpFilter0` and `ExpFilter1`.
 * Below is an example of an action list with two different exponential filters:
 *
 * \snippet DiscontinuousGalerkin/Test_Filtering.cpp action_list_example
 *
 * #### Action properties
 *
 * Uses:
 * - ConstGlobalCache:
 *   - `ExponentialFilter`
 * - DataBox:
 *   - `Tags::Mesh`
 * - DataBox changes:
 *   - Adds: nothing
 *   - Removes: nothing
 *   - Modifies:
 *     - `TagsToFilter`
 * - System:
 *   - `volume_dim`
 *   - `variables_tag`
 *
 * #### Design decision:
 *
 * - The reason for the `size_t` template parameter is to allow for different
 * `Alpha` and `HalfPower` parameters for different tensors while still being
 * able to cache the matrices.
 */
template <size_t FilterIndex, typename... TagsToFilter>
class ExponentialFilter<FilterIndex, tmpl::list<TagsToFilter...>> {
 public:
  using const_global_cache_tags = tmpl::list<ExponentialFilter>;

  /// \brief The value of `exp(-alpha)` is what the highest modal coefficient is
  /// rescaled by.
  struct Alpha {
    using type = double;
    static constexpr OptionString help =
        "exp(-alpha) is rescaling of highest coefficient";
    static type lower_bound() noexcept { return 0.0; }
  };
  /*!
   * \brief Half of the exponent in the exponential.
   *
   * \f{align*}{
   *  c_i\to c_i \exp\left[-\alpha \left(\frac{i}{N}\right)^{2m}\right]
   * \f}
   */
  struct HalfPower {
    using type = unsigned;
    static constexpr OptionString help =
        "Half of the exponent in the generalized Gaussian";
    static type lower_bound() noexcept { return 1; }
  };
  /// \brief Turn the filter off
  ///
  /// This option exists to temporarily disable the filter for debugging
  /// purposes. For problems where filtering is not needed, the preferred
  /// approach is to not compile the filter into the executable.
  struct DisableForDebugging {
    using type = bool;
    static type default_value() noexcept { return false; }
    static constexpr OptionString help = {"Disable the filter"};
  };

  using type = ExponentialFilter;
  static std::string name() noexcept {
    return "ExpFilter" + std::to_string(FilterIndex);
  }

  using options = tmpl::list<Alpha, HalfPower, DisableForDebugging>;
  static constexpr OptionString help = {"An exponential filter."};

  ExponentialFilter() = default;

  ExponentialFilter(double alpha, unsigned half_power,
                    bool disable_for_debugging) noexcept;

  // Action part of the class
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    constexpr size_t volume_dim = Metavariables::system::volume_dim;
    using evolved_vars_tag = typename Metavariables::system::variables_tag;
    using evolved_vars_tags_list = typename evolved_vars_tag::tags_list;
    const ExponentialFilter& filter_helper =
        Parallel::get<ExponentialFilter>(cache);
    if (UNLIKELY(filter_helper.disable_for_debugging_)) {
      return {std::move(box)};
    }
    const Mesh<volume_dim> mesh = db::get<::Tags::Mesh<volume_dim>>(box);
    const Matrix empty{};
    auto filter = make_array<volume_dim>(std::cref(empty));
    for (size_t d = 0; d < volume_dim; d++) {
      gsl::at(filter, d) =
          std::cref(filter_helper.filter_matrix(mesh.slice_through(d)));
    }

    // In the case that the tags we are filtering are all the evolved variables
    // we filter the entire Variables at once to be more efficient. This case is
    // the first branch of the `if-else`.
    if (ExponentialFilter_detail::FilterAllEvolvedVars<
            sizeof...(TagsToFilter) ==
                tmpl::size<evolved_vars_tags_list>::value,
            cpp17::is_same_v<evolved_vars_tags_list,
                             tmpl::list<TagsToFilter...>>>::
            template f<evolved_vars_tags_list, TagsToFilter...>::value) {
      db::mutate<typename Metavariables::system::variables_tag>(
          make_not_null(&box),
          [&filter](
              const gsl::not_null<
                  db::item_type<typename Metavariables::system::variables_tag>*>
                  vars,
              const auto& local_mesh) noexcept {
            *vars = apply_matrices(filter, *vars, local_mesh.extents());
          },
          mesh);
    } else {
      db::mutate<TagsToFilter...>(
          make_not_null(&box),
          [&filter](const gsl::not_null<
                        db::item_type<TagsToFilter>*>... tensors_to_filter,
                    const auto& local_mesh) noexcept {
            DataVector temp(local_mesh.number_of_grid_points(), 0.0);
            const auto helper =
                [&local_mesh, &filter, &temp ](const auto tensor) noexcept {
              for (auto& component : *tensor) {
                temp = 0.0;
                apply_matrices(make_not_null(&temp), filter, component,
                               local_mesh.extents());
                component = temp;
              }
            };
            EXPAND_PACK_LEFT_TO_RIGHT(helper(tensors_to_filter));
          },
          mesh);
    }
    return {std::move(box)};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

  bool operator==(const ExponentialFilter& rhs) const noexcept;

 private:
  const Matrix& filter_matrix(const Mesh<1>& mesh) const noexcept {
    // All these switch gymnastics are to translate the runtime mesh values into
    // compile time template parameters. This is used so we have a sparse lazy
    // cache where we only cache matrices that we actually need for the meshes
    // used in the simulation.
    switch (mesh.basis(0)) {
      case Spectral::Basis::Legendre:
        switch (mesh.quadrature(0)) {
          case Spectral::Quadrature::Gauss:
            return filter_matrix_impl<Spectral::Basis::Legendre,
                                      Spectral::Quadrature::Gauss>(
                mesh,
                std::make_index_sequence<Spectral::maximum_number_of_points<
                    Spectral::Basis::Legendre>>{});
          case Spectral::Quadrature::GaussLobatto:
            return filter_matrix_impl<Spectral::Basis::Legendre,
                                      Spectral::Quadrature::GaussLobatto>(
                mesh,
                std::make_index_sequence<Spectral::maximum_number_of_points<
                    Spectral::Basis::Legendre>>{});
          default:
            ERROR("Missing quadrature in exponential filter matrix action.");
        }
      case Spectral::Basis::Chebyshev:
        switch (mesh.quadrature(0)) {
          case Spectral::Quadrature::Gauss:
            return filter_matrix_impl<Spectral::Basis::Chebyshev,
                                      Spectral::Quadrature::Gauss>(
                mesh,
                std::make_index_sequence<Spectral::maximum_number_of_points<
                    Spectral::Basis::Chebyshev>>{});
          case Spectral::Quadrature::GaussLobatto:
            return filter_matrix_impl<Spectral::Basis::Chebyshev,
                                      Spectral::Quadrature::GaussLobatto>(
                mesh.slice_through(0),
                std::make_index_sequence<Spectral::maximum_number_of_points<
                    Spectral::Basis::Chebyshev>>{});
          default:
            ERROR("Missing quadrature in exponential filter matrix action.");
        }
      default:
        ERROR("Missing basis in exponential filter matrix action.");
    };
  }

  template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType,
            size_t I>
  static const Matrix& filter_matrix_cache(const Mesh<1>& mesh,
                                           const double alpha,
                                           const unsigned half_power) noexcept {
    static Matrix matrix =
        Spectral::filtering::exponential_filter(mesh, alpha, half_power);
    return matrix;
  }

  template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType,
            size_t... Is>
  const Matrix& filter_matrix_impl(const Mesh<1>& mesh,
                                   std::index_sequence<Is...> /*meta*/) const
      noexcept {
    static const std::array<const Matrix& (*)(const Mesh<1>&, double, unsigned),
                            sizeof...(Is)>
        cache{{&filter_matrix_cache<BasisType, QuadratureType, Is>...}};
    return gsl::at(cache, mesh.extents(0))(mesh, alpha_, half_power_);
  }

  double alpha_{36.0};
  unsigned half_power_{16};
  bool disable_for_debugging_{false};
};

template <size_t FilterIndex, typename... TagsToFilter>
ExponentialFilter<FilterIndex, tmpl::list<TagsToFilter...>>::ExponentialFilter(
    const double alpha, const unsigned half_power,
    const bool disable_for_debugging) noexcept
    : alpha_(alpha),
      half_power_(half_power),
      disable_for_debugging_(disable_for_debugging) {}

template <size_t FilterIndex, typename... TagsToFilter>
void ExponentialFilter<FilterIndex, tmpl::list<TagsToFilter...>>::pup(
    PUP::er& p) noexcept {
  p | alpha_;
  p | half_power_;
  p | disable_for_debugging_;
}

template <size_t FilterIndex, typename... TagsToFilter>
bool ExponentialFilter<FilterIndex, tmpl::list<TagsToFilter...>>::operator==(
    const ExponentialFilter& rhs) const noexcept {
  return alpha_ == rhs.alpha_ and half_power_ == rhs.half_power_ and
         disable_for_debugging_ == rhs.disable_for_debugging_;
}

template <size_t FilterIndex, typename TagsToFilterList>
bool operator!=(
    const ExponentialFilter<FilterIndex, TagsToFilterList>& lhs,
    const ExponentialFilter<FilterIndex, TagsToFilterList>& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace Actions
}  // namespace dg
