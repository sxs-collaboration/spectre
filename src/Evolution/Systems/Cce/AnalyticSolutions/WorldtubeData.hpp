// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/Initialize/InverseCubic.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cce {
/// Analytic solutions for CCE worldtube data and corresponding waveform News
namespace Solutions {

/// The collection of cache tags for `WorldtubeData`
using cce_analytic_solutions_cache_tags = tmpl::list<
    Tags::CauchyCartesianCoords, Tags::Dr<Tags::CauchyCartesianCoords>,
    gr::Tags::SpacetimeMetric<DataVector, 3>, gh::Tags::Pi<DataVector, 3>,
    gh::Tags::Phi<DataVector, 3>, gr::Tags::SpatialMetric<DataVector, 3>,
    gr::Tags::Shift<DataVector, 3>, gr::Tags::Lapse<DataVector>,
    ::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>,
    ::Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>,
    ::Tags::dt<gr::Tags::Shift<DataVector, 3>>,
    ::Tags::dt<gr::Tags::Lapse<DataVector>>,
    Tags::Dr<gr::Tags::SpatialMetric<DataVector, 3>>,
    Tags::Dr<gr::Tags::Shift<DataVector, 3>>,
    Tags::Dr<gr::Tags::Lapse<DataVector>>, Tags::News>;

/// The collection of cache tags for `KleinGordonWorldtubeData`
using kg_cce_analytic_solutions_cache_tags =
    tmpl::list<Tags::CauchyCartesianCoords, Tags::KleinGordonPsi,
               Tags::KleinGordonPi>;

/// \cond
class BouncingBlackHole;
class GaugeWave;
class LinearizedBondiSachs;
class RobinsonTrautman;
class RotatingSchwarzschild;
class TeukolskyWave;
/// \endcond

/*!
 * \brief Base class for `WorldtubeData` and `KleinGordonWorldtubeData`
 *
 * \details The tuple `IntermediateCacheTuple` is required by both
 * `WorldtubeData` and `KleinGordonWorldtubeData`. This base class constructs it
 * utilizing the supplied `CacheTagList`.
 */
template <typename CacheTagList>
struct WorldtubeDataBase {
 protected:
  template <typename Tag>
  struct IntermediateCache {
    typename Tag::type data;
    size_t l_max = 0;
    double time = -std::numeric_limits<double>::infinity();
  };

  template <typename Tag>
  struct IntermediateCacheTag {
    using type = IntermediateCache<Tag>;
  };

  using IntermediateCacheTuple =
      tuples::tagged_tuple_from_typelist<tmpl::transform<
          CacheTagList, tmpl::bind<IntermediateCacheTag, tmpl::_1>>>;

  // NOLINTNEXTLINE(spectre-mutable)
  mutable IntermediateCacheTuple intermediate_cache_;
};

/*!
 * \brief Abstract base class for analytic worldtube data for verifying the CCE
 * system.
 *
 * \details All of the boundary data quantities are provided by the
 * `WorldtubeData::variables()` function.
 *
 * This class provides caching and conversion between different
 * representations of the metric data needed for the worldtube computation and
 * evolution. The set of pure virtual functions (required to be overriden in the
 * derived classes) is:
 * - `WorldtubeData::get_clone()`: should return a
 * `std::unique_ptr<WorldtubeData>` with cloned state
 * - `WorldtubeData::variables_impl()` (a `protected` function): should compute
 * and return by `not_null` pointer the spacetime metric quantity requested in
 * the final (metavariable) tag argument. The function overloads that are
 * required to be overriden in the derived class are
 * `gr::Tags::SpacetimeMetric<DataVector, 3>`,
 * `::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>`,
 * `gh::Tags::Phi<DataVector, 3>`, and
 * `Cce::Tags::News`.
 * - `prepare_solution()`: Any initial precomputation needed to determine all of
 * the solutions efficiently. This function is called by the base class prior to
 * computing or retrieving from the internal cache the requested quantities.
 *
 * \warning This class is not intended to be threadsafe! Therefore, using
 * instances of this class placed into the const global cache results in
 * undefined behavior. The analytic data for CCE is not easily represented as a
 * full closed-form solution for the entire Bondi-Sachs-like metric over the
 * domain, so this class and its descendants perform numerical calculations such
 * as spin-weighted derivatives over the sphere. Instead, it makes best sense to
 * compute the global solution over the extraction sphere, and cache
 * intermediate steps to avoid repeating potentially expensive tensor
 * calculations.
 */
struct WorldtubeData
    : public PUP::able,
      public WorldtubeDataBase<cce_analytic_solutions_cache_tags> {
  using creatable_classes =
      tmpl::list<BouncingBlackHole, GaugeWave, LinearizedBondiSachs,
                 RobinsonTrautman, RotatingSchwarzschild, TeukolskyWave>;

  /// The set of available tags provided by the analytic solution
  using tags = tmpl::list<
      Tags::CauchyCartesianCoords, Tags::Dr<Tags::CauchyCartesianCoords>,
      gr::Tags::SpacetimeMetric<DataVector, 3>,
      ::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>,
      gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>,
      gr::Tags::SpatialMetric<DataVector, 3>,
      ::Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>,
      Tags::Dr<gr::Tags::SpatialMetric<DataVector, 3>>,
      gr::Tags::Shift<DataVector, 3>,
      ::Tags::dt<gr::Tags::Shift<DataVector, 3>>,
      Tags::Dr<gr::Tags::Shift<DataVector, 3>>, gr::Tags::Lapse<DataVector>,
      ::Tags::dt<gr::Tags::Lapse<DataVector>>,
      Tags::Dr<gr::Tags::Lapse<DataVector>>, Tags::News>;

  WRAPPED_PUPable_abstract(WorldtubeData);  // NOLINT

  // clang doesn't manage to use = default correctly in this case
  // NOLINTNEXTLINE(modernize-use-equals-default)
  WorldtubeData() {}

  explicit WorldtubeData(const double extraction_radius)
      : extraction_radius_{extraction_radius} {}

  explicit WorldtubeData(CkMigrateMessage* msg) : PUP::able(msg) {}

  ~WorldtubeData() override = default;

  virtual std::unique_ptr<WorldtubeData> get_clone() const = 0;

  /*!
   * \brief Retrieve worldtube data represented by the analytic solution, at
   * boundary angular resolution `l_max` and time `time`
   *
   * \details The set of requested tags are specified by the final argument,
   * which must be a `tmpl::list` of tags to be retrieved. The set of available
   * tags is found in `WorldtubeData::tags`, and includes coordinate and
   * Jacobian quantities as well as metric quantities and derivatives thereof.
   */
  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const size_t output_l_max, const double time,
      tmpl::list<Tags...> /*meta*/) const {
    prepare_solution(output_l_max, time);
    return {cache_or_compute<Tags>(output_l_max, time)...};
  }

  void pup(PUP::er& p) override;

  virtual std::unique_ptr<Cce::InitializeJ::InitializeJ<false>>
  get_initialize_j(const double /*start_time*/) const {
    return std::make_unique<Cce::InitializeJ::InverseCubic<false>>();
  };

  virtual bool use_noninertial_news() const { return false; }

 protected:
  template <typename Tag>
  const auto& cache_or_compute(const size_t output_l_max,
                               const double time) const {
    auto& item_cache = get<IntermediateCacheTag<Tag>>(intermediate_cache_);
    if (item_cache.l_max == output_l_max and item_cache.time == time) {
      return item_cache.data;
    }
    auto& item = item_cache.data;
    set_number_of_grid_points(
        make_not_null(&item),
        Spectral::Swsh::number_of_swsh_collocation_points(output_l_max));
    variables_impl(make_not_null(&item), output_l_max, time,
                   tmpl::type_<Tag>{});
    item_cache.l_max = output_l_max;
    item_cache.time = time;
    return item;
  }
  virtual void prepare_solution(size_t output_l_max, double time) const = 0;

  // note that function template cannot be virtual, so we have to emulate
  // template specializations through function overloads
  virtual void variables_impl(
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_coordinates,
      size_t output_l_max, double time,
      tmpl::type_<Tags::CauchyCartesianCoords> /*meta*/) const;

  virtual void variables_impl(
      gsl::not_null<tnsr::i<DataVector, 3>*> dr_cartesian_coordinates,
      size_t output_l_max, double time,
      tmpl::type_<Tags::Dr<Tags::CauchyCartesianCoords>> /*meta*/) const;

  virtual void variables_impl(
      gsl::not_null<tnsr::aa<DataVector, 3>*> spacetime_metric,
      size_t output_l_max, double time,
      tmpl::type_<gr::Tags::SpacetimeMetric<DataVector, 3>>
      /*meta*/) const = 0;

  virtual void variables_impl(
      gsl::not_null<tnsr::aa<DataVector, 3>*> dt_spacetime_metric,
      size_t output_l_max, double time,
      tmpl::type_<::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>>
      /*meta*/) const = 0;

  virtual void variables_impl(gsl::not_null<tnsr::aa<DataVector, 3>*> pi,
                              size_t output_l_max, double time,
                              tmpl::type_<gh::Tags::Pi<DataVector, 3>>
                              /*meta*/) const;

  virtual void variables_impl(
      gsl::not_null<tnsr::iaa<DataVector, 3>*> d_spacetime_metric,
      size_t output_l_max, double time,
      tmpl::type_<gh::Tags::Phi<DataVector, 3>>
      /*meta*/) const = 0;

  virtual void variables_impl(
      gsl::not_null<tnsr::ii<DataVector, 3>*> spatial_metric,
      size_t output_l_max, double time,
      tmpl::type_<gr::Tags::SpatialMetric<DataVector, 3>>
      /*meta*/) const;

  virtual void variables_impl(
      gsl::not_null<tnsr::ii<DataVector, 3>*> dt_spatial_metric,
      size_t output_l_max, double time,
      tmpl::type_<::Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>>
      /*meta*/) const;

  virtual void variables_impl(
      gsl::not_null<tnsr::ii<DataVector, 3>*> dr_spatial_metric,
      size_t output_l_max, double time,
      tmpl::type_<Tags::Dr<gr::Tags::SpatialMetric<DataVector, 3>>>
      /*meta*/) const;

  virtual void variables_impl(gsl::not_null<tnsr::I<DataVector, 3>*> shift,
                              size_t output_l_max, double time,
                              tmpl::type_<gr::Tags::Shift<DataVector, 3>>
                              /*meta*/) const;

  virtual void variables_impl(
      gsl::not_null<tnsr::I<DataVector, 3>*> dt_shift, size_t output_l_max,
      double time, tmpl::type_<::Tags::dt<gr::Tags::Shift<DataVector, 3>>>
      /*meta*/) const;

  virtual void variables_impl(
      gsl::not_null<tnsr::I<DataVector, 3>*> dr_shift, size_t output_l_max,
      double time, tmpl::type_<Tags::Dr<gr::Tags::Shift<DataVector, 3>>>
      /*meta*/) const;

  virtual void variables_impl(gsl::not_null<Scalar<DataVector>*> lapse,
                              size_t output_l_max, double time,
                              tmpl::type_<gr::Tags::Lapse<DataVector>>
                              /*meta*/) const;

  virtual void variables_impl(
      gsl::not_null<Scalar<DataVector>*> dt_lapse, size_t output_l_max,
      double time, tmpl::type_<::Tags::dt<gr::Tags::Lapse<DataVector>>>
      /*meta*/) const;

  virtual void variables_impl(gsl::not_null<Scalar<DataVector>*> dr_lapse,
                              size_t output_l_max, double time,
                              tmpl::type_<Tags::Dr<gr::Tags::Lapse<DataVector>>>
                              /*meta*/) const;

  virtual void variables_impl(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> news,
      size_t output_l_max, double time,
      tmpl::type_<Tags::News> /*meta*/) const = 0;

  double extraction_radius_ = std::numeric_limits<double>::quiet_NaN();
};

/*!
 * \brief Abstract base class for analytic worldtube data for verifying the
 * Klein-Gordon CCE system.
 *
 * \details All of the boundary data quantities are provided by the
 * `KleinGordonWorldtubeData::variables()` function.
 *
 * This class provides caching and conversion between different
 * representations of the scalar data needed for the worldtube computation and
 * evolution. The set of pure virtual functions (required to be overriden in the
 * derived classes) is:
 * - `KleinGordonWorldtubeData::get_clone()`: should return a
 * `std::unique_ptr<KleinGordonWorldtubeData>` with cloned state
 * - `KleinGordonWorldtubeData::variables_impl()` (a `protected` function):
 * should compute and return by `not_null` pointer the scalar quantity
 * requested in the final (metavariable) tag argument. The function overloads
 * that are required to be overriden in the derived class are
 * `Cce::Tags::KleinGordonPsi`,
 * `Cce::Tags::KleinGordonPi`, and
 * `Cce::Tags::CauchyCartesianCoords`.
 * - `prepare_solution()`: Any initial precomputation needed to determine all of
 * the solutions efficiently. This function is called by the base class prior to
 * computing or retrieving from the internal cache the requested quantities.
 *
 * \warning This class is not intended to be threadsafe! Therefore, using
 * instances of this class placed into the const global cache results in
 * undefined behavior. The analytic data for CCE is not easily represented as a
 * full closed-form solution for the scalar field over the
 * domain, so this class and its descendants perform numerical calculations.
 * Instead, it makes best sense to compute the global solution over the
 * extraction sphere, and cache intermediate steps to avoid repeating
 * potentially expensive calculations.
 */
struct KleinGordonWorldtubeData
    : public PUP::able,
      WorldtubeDataBase<kg_cce_analytic_solutions_cache_tags> {
  using creatable_classes = tmpl::list<>;

  /// The set of available tags provided by the analytic solution
  using tags = tmpl::list<Tags::CauchyCartesianCoords, Tags::KleinGordonPsi,
                          Tags::KleinGordonPi>;

  WRAPPED_PUPable_abstract(KleinGordonWorldtubeData);  // NOLINT

  // clang doesn't manage to use = default correctly in this case
  // NOLINTNEXTLINE(modernize-use-equals-default)
  KleinGordonWorldtubeData() {}

  explicit KleinGordonWorldtubeData(const double extraction_radius)
      : extraction_radius_{extraction_radius} {}

  explicit KleinGordonWorldtubeData(CkMigrateMessage* msg) : PUP::able(msg) {}

  ~KleinGordonWorldtubeData() override = default;

  virtual std::unique_ptr<KleinGordonWorldtubeData> get_clone() const = 0;

  /*!
   * \brief Retrieve worldtube data represented by the analytic solution, at
   * boundary angular resolution `l_max` and time `time`
   *
   * \details The set of requested tags are specified by the final argument,
   * which must be a `tmpl::list` of tags to be retrieved. The set of available
   * tags is found in `KleinGordonWorldtubeData::tags`.
   */
  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const size_t output_l_max, const double time,
      tmpl::list<Tags...> /*meta*/) const {
    prepare_solution(output_l_max, time);
    return {cache_or_compute<Tags>(output_l_max, time)...};
  }

  void pup(PUP::er& p) override;

 protected:
  template <typename Tag>
  const auto& cache_or_compute(const size_t output_l_max,
                               const double time) const {
    auto& item_cache = get<IntermediateCacheTag<Tag>>(intermediate_cache_);
    if (item_cache.l_max == output_l_max and item_cache.time == time) {
      return item_cache.data;
    }
    auto& item = item_cache.data;
    set_number_of_grid_points(
        make_not_null(&item),
        Spectral::Swsh::number_of_swsh_collocation_points(output_l_max));
    variables_impl(make_not_null(&item), output_l_max, time,
                   tmpl::type_<Tag>{});
    item_cache.l_max = output_l_max;
    item_cache.time = time;
    return item;
  }
  virtual void prepare_solution(size_t output_l_max, double time) const = 0;

  // note that function template cannot be virtual, so we have to emulate
  // template specializations through function overloads
  virtual void variables_impl(
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_coordinates,
      size_t output_l_max, double time,
      tmpl::type_<Tags::CauchyCartesianCoords> /*meta*/) const;

  virtual void variables_impl(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> kg_psi,
      size_t output_l_max, double time,
      tmpl::type_<Tags::KleinGordonPsi> /*meta*/) const = 0;

  virtual void variables_impl(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> kg_pi,
      size_t output_l_max, double time,
      tmpl::type_<Tags::KleinGordonPi> /*meta*/) const = 0;

  double extraction_radius_ = std::numeric_limits<double>::quiet_NaN();
};
}  // namespace Solutions
}  // namespace Cce
