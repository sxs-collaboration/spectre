// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/SpecBoundaryData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/WriteSimpleData.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/MakeString.hpp"
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
  AnalyticBoundaryDataManager() noexcept = default;

  AnalyticBoundaryDataManager(
      size_t l_max, double extraction_radius,
      std::unique_ptr<Solutions::WorldtubeData> generator) noexcept;

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
      double time) const noexcept;

  /// Use `observers::ThreadedActions::WriteSimpleData` to output the expected
  /// news at `time` from the analytic data to dataset `/expected_news.dat`.
  template <typename Metavariables>
  void write_news(Parallel::GlobalCache<Metavariables>& cache,
                  double time) const noexcept;

  size_t get_l_max() const noexcept { return l_max_; }

  /// Serialization for Charm++.
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

 private:
  size_t l_max_ = 0;
  std::unique_ptr<Solutions::WorldtubeData> generator_;
  double extraction_radius_ = std::numeric_limits<double>::signaling_NaN();
};

template <typename Metavariables>
void AnalyticBoundaryDataManager::write_news(
    Parallel::GlobalCache<Metavariables>& cache,
    const double time) const noexcept {
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
      observers::ObserverWriter<Metavariables>>(
      cache)[static_cast<size_t>(sys::my_node())];
  Parallel::threaded_action<observers::ThreadedActions::WriteSimpleData>(
      observer_proxy, file_legend, data_to_write, "/expected_news"s);
}
}  // namespace Cce
