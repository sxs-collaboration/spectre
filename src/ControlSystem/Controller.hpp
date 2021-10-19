// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "Options/Options.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// \ingroup ControlSystemGroup
/// A PND (proportional to Q and N derivatives of Q) controller that computes
/// the control signal:
/// \f[ U(t) = \sum_{k=0}^{N} a_{k} \frac{d^kQ}{dt^k} \f]
/// where N is specified by the template parameter `DerivOrder`.
///
/// If an averager is used for `q_and_derivs` (as we typically do), there is an
/// induced time offset, \f$\Delta t\f$, due to the time-weighted averaging.
/// Therefore, the `q_and_derivs` that we have in hand are at some time
/// \f$t_{0}\f$. However, we desire `q_and_derivs` at the current time
/// \f$t = t_{0} + \Delta t\f$ to determine the appropriate control
/// signal. We accomplish this by Taylor expanding
/// \f$Q(t_{0} + \Delta t)\f$. The averager allows for averaging of
/// \f$Q\f$ and its derivatives OR to not average \f$Q\f$ while still averaging
/// the derivatives (the derivatives are always averaged in order to reduce
/// noise due to numerical differentiation). When they are both averaged, the
/// time offset will be identical for \f$Q\f$ and the derivatives,
/// i.e. `q_time_offset` = `deriv_time_offset`. If an unaveraged \f$Q\f$ is
/// used, then the time offset associated with \f$Q\f$ is zero,
/// i.e. `q_time_offset`=0. and the derivative time offset, `deriv_time_offset`,
/// remains non-zero.
template <size_t DerivOrder>
class Controller {
 public:
  struct UpdateFraction {
    using type = double;
    static constexpr Options::String help = {
        "Fraction of damping timescale used to determine how often to update "
        "functions of time."};
  };

  using options = tmpl::list<UpdateFraction>;
  static constexpr Options::String help{
      "Computes control signal used to reset highest derivative of a function "
      "of time. Also determines when a function of time needs to be updated "
      "next."};

  Controller(const double update_fraction)
      : update_fraction_(update_fraction) {}

  Controller() = default;
  Controller(Controller&&) = default;
  Controller& operator=(Controller&&) = default;
  Controller(const Controller&) = default;
  Controller& operator=(const Controller&) = default;
  ~Controller() = default;

  DataVector operator()(
      const DataVector& timescales,
      const std::array<DataVector, DerivOrder + 1>& q_and_derivs,
      double q_time_offset, double deriv_time_offset) const;

  /// Takes the current minimum of all timescales and uses that to set the time
  /// between updates
  void assign_time_between_updates(const double current_min_timescale) {
    time_between_updates_ = update_fraction_ * current_min_timescale;
  }

  double get_update_fraction() const { return update_fraction_; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) {
    p | update_fraction_;
    p | time_between_updates_;
  }

  template <size_t LocalDerivOrder>
  // NOLINTNEXTLINE(readability-redundant-declaration) false positive
  friend bool operator==(const Controller<LocalDerivOrder>& lhs,
                         const Controller<LocalDerivOrder>& rhs);

 private:
  // If update_fraction_ isn't set we need to error
  double update_fraction_{std::numeric_limits<double>::signaling_NaN()};
  // If this time_between_triggers_ isn't set, the default should just be that
  // the functions of time are never updated (i.e. infinity)
  double time_between_updates_{std::numeric_limits<double>::infinity()};
};

template <size_t DerivOrder>
bool operator!=(const Controller<DerivOrder>& lhs,
                const Controller<DerivOrder>& rhs);
