// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cce {
namespace Solutions {

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
 * `gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>`,
 * `::Tags::dt<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>`,
 * `GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>`, and
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
struct WorldtubeData : public PUP::able {
  using creatable_classes = tmpl::list<>;

  /// The set of available tags provided by the analytic solution
  using tags = tmpl::list<
      Tags::CauchyCartesianCoords, Tags::Dr<Tags::CauchyCartesianCoords>,
      gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>,
      ::Tags::dt<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>,
      GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>,
      GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>,
      gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>,
      ::Tags::dt<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>,
      Tags::Dr<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>,
      gr::Tags::Shift<3, ::Frame::Inertial, DataVector>,
      ::Tags::dt<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>,
      Tags::Dr<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>,
      gr::Tags::Lapse<DataVector>, ::Tags::dt<gr::Tags::Lapse<DataVector>>,
      Tags::Dr<gr::Tags::Lapse<DataVector>>, Tags::News>;

  WRAPPED_PUPable_abstract(WorldtubeData);  // NOLINT

  virtual std::unique_ptr<WorldtubeData> get_clone() const noexcept = 0;

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
  tuples::TaggedTuple<Tags...> variables(const size_t output_l_max,
                                         const double time,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    prepare_solution(output_l_max, time);
    return {cache_or_compute<Tags>(output_l_max, time)...};
  }

 protected:
  template <typename Tag>
  const auto& cache_or_compute(const size_t output_l_max,
                               const double time) const noexcept {
    auto& item_cache = get<IntermediateCacheTag<Tag>>(intermediate_cache_);
    if (item_cache.l_max == output_l_max and item_cache.time == time) {
      return item_cache.data;
    }
    auto& item = item_cache.data;
    destructive_resize_components(
        make_not_null(&item),
        Spectral::Swsh::number_of_swsh_collocation_points(output_l_max));
    variables_impl(make_not_null(&item), output_l_max, time,
                   tmpl::type_<Tag>{});
    item_cache.l_max = output_l_max;
    item_cache.time = time;
    return item;
  }
  virtual void prepare_solution(size_t output_l_max, double time) const
      noexcept = 0;

  // note that function template cannot be virtual, so we have to emulate
  // template specializations through function overloads
  virtual void variables_impl(
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_coordinates,
      size_t output_l_max, double time,
      tmpl::type_<Tags::CauchyCartesianCoords> /*meta*/) const noexcept;

  virtual void variables_impl(
      gsl::not_null<tnsr::i<DataVector, 3>*> dr_cartesian_coordinates,
      size_t output_l_max, double time,
      tmpl::type_<Tags::Dr<Tags::CauchyCartesianCoords>> /*meta*/) const
      noexcept;

  virtual void variables_impl(
      gsl::not_null<tnsr::aa<DataVector, 3>*> spacetime_metric,
      size_t output_l_max, double time,
      tmpl::type_<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>
      /*meta*/) const noexcept = 0;

  virtual void variables_impl(
      gsl::not_null<tnsr::aa<DataVector, 3>*> dt_spacetime_metric,
      size_t output_l_max, double time,
      tmpl::type_<::Tags::dt<
          gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>>
      /*meta*/) const noexcept = 0;

  virtual void variables_impl(
      gsl::not_null<tnsr::aa<DataVector, 3>*> pi, size_t output_l_max,
      double time,
      tmpl::type_<GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>>
      /*meta*/) const noexcept;

  virtual void variables_impl(
      gsl::not_null<tnsr::iaa<DataVector, 3>*> d_spacetime_metric,
      size_t output_l_max, double time,
      tmpl::type_<GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>
      /*meta*/) const noexcept = 0;

  virtual void variables_impl(
      gsl::not_null<tnsr::ii<DataVector, 3>*> spatial_metric,
      size_t output_l_max, double time,
      tmpl::type_<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>
      /*meta*/) const noexcept;

  virtual void variables_impl(
      gsl::not_null<tnsr::ii<DataVector, 3>*> dt_spatial_metric,
      size_t output_l_max, double time,
      tmpl::type_<
          ::Tags::dt<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>>
      /*meta*/) const noexcept;

  virtual void variables_impl(
      gsl::not_null<tnsr::ii<DataVector, 3>*> dr_spatial_metric,
      size_t output_l_max, double time,
      tmpl::type_<
          Tags::Dr<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>>
      /*meta*/) const noexcept;

  virtual void variables_impl(
      gsl::not_null<tnsr::I<DataVector, 3>*> shift, size_t output_l_max,
      double time,
      tmpl::type_<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>
      /*meta*/) const noexcept;

  virtual void variables_impl(
      gsl::not_null<tnsr::I<DataVector, 3>*> dt_shift, size_t output_l_max,
      double time,
      tmpl::type_<::Tags::dt<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>>
      /*meta*/) const noexcept;

  virtual void variables_impl(
      gsl::not_null<tnsr::I<DataVector, 3>*> dr_shift, size_t output_l_max,
      double time,
      tmpl::type_<Tags::Dr<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>>
      /*meta*/) const noexcept;

  virtual void variables_impl(gsl::not_null<Scalar<DataVector>*> lapse,
                              size_t output_l_max, double time,
                              tmpl::type_<gr::Tags::Lapse<DataVector>>
                              /*meta*/) const noexcept;

  virtual void variables_impl(
      gsl::not_null<Scalar<DataVector>*> dt_lapse, size_t output_l_max,
      double time, tmpl::type_<::Tags::dt<gr::Tags::Lapse<DataVector>>>
      /*meta*/) const noexcept;

  virtual void variables_impl(gsl::not_null<Scalar<DataVector>*> dr_lapse,
                              size_t output_l_max, double time,
                              tmpl::type_<Tags::Dr<gr::Tags::Lapse<DataVector>>>
                              /*meta*/) const noexcept;

  virtual void variables_impl(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> news,
      size_t output_l_max, double time, tmpl::type_<Tags::News> /*meta*/) const
      noexcept = 0;

  template <typename Tag>
  struct IntermediateCache {
    db::item_type<Tag> data;
    size_t l_max = 0;
    double time = -std::numeric_limits<double>::infinity();
  };

  template <typename Tag>
  struct IntermediateCacheTag {
    using type = IntermediateCache<Tag>;
  };

  using IntermediateCacheTuple =
      tuples::tagged_tuple_from_typelist<tmpl::transform<
          tmpl::list<
              Tags::CauchyCartesianCoords,
              Tags::Dr<Tags::CauchyCartesianCoords>,
              gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>,
              GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>,
              GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>,
              gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>,
              gr::Tags::Shift<3, ::Frame::Inertial, DataVector>,
              gr::Tags::Lapse<DataVector>,
              ::Tags::dt<
                  gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>,
              ::Tags::dt<
                  gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>,
              ::Tags::dt<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>,
              ::Tags::dt<gr::Tags::Lapse<DataVector>>,
              Tags::Dr<
                  gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>,
              Tags::Dr<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>,
              Tags::Dr<gr::Tags::Lapse<DataVector>>, Tags::News>,
          tmpl::bind<IntermediateCacheTag, tmpl::_1>>>;

  mutable IntermediateCacheTuple intermediate_cache_;
  double extraction_radius_ = std::numeric_limits<double>::quiet_NaN();
};
}  // namespace Solutions
}  // namespace Cce
