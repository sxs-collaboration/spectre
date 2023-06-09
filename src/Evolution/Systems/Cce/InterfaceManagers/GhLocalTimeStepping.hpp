// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <tuple>

#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhInterfaceManager.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::InterfaceManagers {

/*!
 * \brief Implementation of a `GhInterfaceManager` that provides data according
 * to interpolation between provided GH data.
 *
 * \details This class receives data from the Generalized Harmonic system
 * sufficient to perform interpolation to arbitrary times required by CCE. From
 * the Generalized Harmonic system, it receives the spacetime metric \f$g_{a
 * b}\f$ and Generalized Harmonic \f$\Phi_{i a b}\f$ and \f$\Pi_{ab}\f$ and the
 * current time via `GhLocalTimeStepping::insert_gh_data()`. The CCE
 * system supplies requests for time steps via
 * `GhLocalTimeStepping::request_gh_data()` and receives interpolated boundary
 * data via `GhLocalTimeStepping::retrieve_and_remove_first_ready_gh_data()`.
 */
class GhLocalTimeStepping : public GhInterfaceManager {
 public:
  struct BoundaryInterpolator {
    using type = std::unique_ptr<intrp::SpanInterpolator>;
    static constexpr Options::String help = {
        "Interpolator for computing CCE data from GH time points"};
  };

  static constexpr Options::String help{
      "Interpolate data from the GH system to generate CCE inputs"};

  using options = tmpl::list<BoundaryInterpolator>;

  GhLocalTimeStepping() = default;
  GhLocalTimeStepping(const GhLocalTimeStepping& rhs)
      : gh_data_{rhs.gh_data_},
        requests_{rhs.requests_},
        latest_removed_{rhs.latest_removed_} {
    if (rhs.interpolator_.get() != nullptr) {
      interpolator_ = rhs.interpolator_->get_clone();
    }
  }

  explicit GhLocalTimeStepping(
      std::unique_ptr<intrp::SpanInterpolator> interpolator)
      : interpolator_{std::move(interpolator)} {}
  GhLocalTimeStepping& operator=(const GhLocalTimeStepping& rhs) {
    if (rhs.interpolator_.get() != nullptr) {
      interpolator_ = rhs.interpolator_->get_clone();
    }
    gh_data_ = rhs.gh_data_;
    requests_ = rhs.requests_;
    latest_removed_ = rhs.latest_removed_;
    return *this;
  }

  explicit GhLocalTimeStepping(CkMigrateMessage* /*unused*/) {}

  WRAPPED_PUPable_decl_template(GhLocalTimeStepping);  // NOLINT

  std::unique_ptr<GhInterfaceManager> get_clone() const override;

  /// \brief Store the provided data set to prepare for interpolation.
  void insert_gh_data(const LinkedMessageId<double>& time_and_previous,
                      const tnsr::aa<DataVector, 3>& spacetime_metric,
                      const tnsr::iaa<DataVector, 3>& phi,
                      const tnsr::aa<DataVector, 3>& pi);

  /// \brief Store the next time step that will be required by the CCE system to
  /// proceed with the evolution.
  void request_gh_data(const TimeStepId& time_id) override;

  /// \brief Return a `std::optional` of either the dense-output data at the
  /// least recently requested time, or `std::nullopt` if not enough GH data has
  /// been supplied yet.
  auto retrieve_and_remove_first_ready_gh_data()
      -> std::optional<std::tuple<TimeStepId, gh_variables>> override;

  /// The number of requests that have been submitted and not yet retrieved.
  size_t number_of_pending_requests() const override {
    return requests_.size();
  }

  /// \brief  The number of times for which data from the GH system is stored.
  size_t number_of_gh_times() const override { return gh_data_.size(); }

  /// Serialization for Charm++.
  void pup(PUP::er& p) override;

 private:
  void clean_up_gh_data();

  std::map<::LinkedMessageId<double>, gh_variables,
           LinkedMessageIdLessComparator<double>>
      gh_data_;
  std::set<TimeStepId> requests_;
  std::unique_ptr<intrp::SpanInterpolator> interpolator_;
  std::optional<double> latest_removed_ = std::nullopt;
};
}  // namespace Cce::InterfaceManagers
