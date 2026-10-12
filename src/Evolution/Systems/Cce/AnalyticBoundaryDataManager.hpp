// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCoefficients.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTransform.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/System/ParallelInfo.hpp"

namespace Cce {
namespace Tags {
/// \cond
struct ObservationLMax;
/// \endcond
}  // namespace Tags

/// A boundary data manager that constructs the desired boundary data into
/// the `Variables` from the data provided by the analytic solution.
class AnalyticBoundaryDataManager {
 public:
  // charm needs an empty constructor.
  AnalyticBoundaryDataManager() = default;

  AnalyticBoundaryDataManager(
      size_t l_max, double extraction_radius,
      std::unique_ptr<Solutions::WorldtubeData> generator,
      double du_dr_j_stencil_step = 0.5);

  /*!
   * \brief Update the `boundary_data_variables` entries for all tags in
   * `Tags::characteristic_worldtube_boundary_tags` to the boundary data from
   * the analytic solution at  `time`.
   *
   * \details This class retrieves metric boundary data from the
   * `Cce::Solutions::WorldtubeData` derived class that represents an analytic
   * solution, then dispatches to `Cce::create_bondi_boundary_data()` to
   * construct the Bondi values into the provided `Variables`
   */
  bool populate_hypersurface_boundary_data(
      gsl::not_null<Variables<
          Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
          boundary_data_variables,
      double time) const;

  /// Use `observers::ThreadedActions::WriteSimpleData` to output the expected
  /// news at `time` from the analytic data to dataset `/expected_news.dat`.
  template <typename ParallelComponent, typename Metavariables>
  void write_news(Parallel::GlobalCache<Metavariables>& cache,
                  double time) const;

  /*!
   * \brief The spacing (in units of the worldtube time) of the time stencil
   * on which the analytic solution is sampled to build the boundary value of
   * `Du<Dr<BondiJ>>`.
   *
   * \details Choose a finite, positive spacing that resolves the shortest
   * timescale of the analytic solution. Too small a spacing amplifies
   * roundoff, while too large a spacing causes truncation error or aliasing
   * that increasing the interpolation order cannot fix. The spacing can be
   * configured with the `Cce: DuDrJStencilStep` option.
   */
  double get_du_dr_j_stencil_step() const { return du_dr_j_stencil_step_; }

  /*!
   * \brief Build the boundary value of `Du<Dr<BondiJ>>` with `interpolator`
   * in `populate_hypersurface_boundary_data`.
   *
   * \details `create_bondi_boundary_data` does not produce that boundary
   * value, and unlike `BondiWorldtubeDataManager` this manager has no stored
   * time series of `Dr<BondiJ>` to differentiate. Instead, it evaluates
   * `Dr<BondiJ>` from the analytic solution on a uniform stencil of
   * `2 * interpolator->required_number_of_points_before_and_after()` times
   * spaced by `get_du_dr_j_stencil_step()` and centered on the requested time.
   * It differentiates this stencil with `interpolator`. As in
   * `BondiWorldtubeDataManager`, the derivative is taken along the worldtube.
   *
   * Without an interpolator (the default, or after passing `nullptr`) the
   * boundary value of `Du<Dr<BondiJ>>` is left untouched, which avoids the
   * additional evaluations of the analytic solution when no initial-data
   * generator needs it.
   *
   * Robinson-Trautman data does not support this centered stencil: requests
   * for past times restart its internal evolution, and times before its
   * initial time are outside its evolved interval. Supplying a non-null
   * interpolator for that solution is an error.
   */
  void set_du_dr_j_interpolator(
      std::unique_ptr<intrp::SpanInterpolator> interpolator);

  size_t get_l_max() const { return l_max_; }

  const Solutions::WorldtubeData& get_generator() const { return *generator_; }

  /// Serialization for Charm++.
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  size_t l_max_ = 0;
  std::unique_ptr<Solutions::WorldtubeData> generator_;
  double extraction_radius_ = std::numeric_limits<double>::signaling_NaN();
  std::unique_ptr<intrp::SpanInterpolator> du_dr_j_interpolator_;
  double du_dr_j_stencil_step_ = 0.5;
};

template <typename ParallelComponent, typename Metavariables>
void AnalyticBoundaryDataManager::write_news(
    Parallel::GlobalCache<Metavariables>& cache, const double time) const {
  const auto news = get<Tags::News>(
      generator_->variables(l_max_, time, tmpl::list<Tags::News>{}));
  const size_t observation_l_max = Parallel::get<Tags::ObservationLMax>(cache);
  std::vector<double> data_to_write(2 * square(observation_l_max + 1) + 1);
  std::vector<std::string> file_legend;
  file_legend.reserve(2 * square(observation_l_max + 1) + 1);
  file_legend.emplace_back("time");
  for (int i = 0; i <= static_cast<int>(observation_l_max); ++i) {
    for (int j = -i; j <= i; ++j) {
      file_legend.push_back(MakeString{} << "Real Y_" << i << "," << j);
      file_legend.push_back(MakeString{} << "Imag Y_" << i << "," << j);
    }
  }
  const ComplexModalVector goldberg_modes =
      Spectral::Swsh::libsharp_to_goldberg_modes(
          Spectral::Swsh::swsh_transform(l_max_, 1, get(news)), l_max_)
          .data();
  data_to_write[0] = time;
  for (size_t i = 0; i < square(observation_l_max + 1); ++i) {
    data_to_write[2 * i + 1] = real(goldberg_modes[i]);
    data_to_write[2 * i + 2] = imag(goldberg_modes[i]);
  }
  auto observer_proxy = Parallel::get_parallel_component<
      observers::ObserverWriter<Metavariables>>(cache)[0];
  const std::string prefix =
      generator_->use_noninertial_news() ? "Noninertial_" : "";
  Parallel::threaded_action<observers::ThreadedActions::WriteReductionDataRow>(
      observer_proxy, "/Cce/News_" + prefix + "expected"s, file_legend,
      std::make_tuple(data_to_write));
}
}  // namespace Cce
