// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <string>

#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace evolution::dg::subcell {
/*!
 * \brief Holds the system-agnostic subcell parameters, such as numbers
 * controlling when to switch between DG and subcell.
 */
class SubcellOptions {
 public:
  struct InitialData {
    static constexpr Options::String help =
        "Parameters only used when setting up initial data.";
  };

  /// The \f$\delta_0\f$ parameter in the relaxed discrete maximum principle
  /// troubled-cell indicator when applied to the initial data
  struct InitialDataRdmpDelta0 {
    static std::string name() { return "RdmpDelta0"; }
    static constexpr Options::String help{"Absolute jump tolerance parameter."};
    using type = double;
    static type lower_bound() { return 0.0; }
    using group = InitialData;
  };
  /// The \f$\epsilon\f$ parameter in the relaxed discrete maximum principle
  /// troubled-cell indicator when applied to the initial data
  struct InitialDataRdmpEpsilon {
    static std::string name() { return "RdmpEpsilon"; }
    static constexpr Options::String help{
        "The jump-dependent relaxation constant."};
    using type = double;
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 1.0; }
    using group = InitialData;
  };
  /// The exponent \f$\alpha\f$ passed to the Persson troubled-cell indicator
  /// when applied to the initial data.
  struct InitialDataPerssonExponent {
    static std::string name() { return "PerssonExponent"; }
    static constexpr Options::String help{
        "The exponent at which the error should decrease with N."};
    using type = double;
    static constexpr type lower_bound() { return 1.0; }
    static constexpr type upper_bound() { return 10.0; }
    using group = InitialData;
  };

  /// The \f$\delta_0\f$ parameter in the relaxed discrete maximum principle
  /// troubled-cell indicator
  struct RdmpDelta0 {
    static std::string name() { return "RdmpDelta0"; }
    static constexpr Options::String help{"Absolute jump tolerance parameter."};
    using type = double;
    static type lower_bound() { return 0.0; }
  };
  /// The \f$\epsilon\f$ parameter in the relaxed discrete maximum principle
  /// troubled-cell indicator
  struct RdmpEpsilon {
    static std::string name() { return "RdmpEpsilon"; }
    static constexpr Options::String help{
        "The jump-dependent relaxation constant."};
    using type = double;
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 1.0; }
  };
  /// The exponent \f$\alpha\f$ passed to the Persson troubled-cell indicator
  struct PerssonExponent {
    static std::string name() { return "PerssonExponent"; }
    static constexpr Options::String help{
        "The exponent at which the error should decrease with N."};
    using type = double;
    static constexpr type lower_bound() { return 1.0; }
    static constexpr type upper_bound() { return 10.0; }
  };
  /// If true, then we always use the subcell method, not DG.
  struct AlwaysUseSubcells {
    static constexpr Options::String help{
        "If true, then always use the subcell method (e.g. finite-difference) "
        "instead of DG."};
    using type = bool;
  };

  using options =
      tmpl::list<InitialDataRdmpDelta0, InitialDataRdmpEpsilon, RdmpDelta0,
                 RdmpEpsilon, InitialDataPerssonExponent, PerssonExponent,
                 AlwaysUseSubcells>;

  static constexpr Options::String help{
      "System-agnostic options for the DG-subcell method."};

  SubcellOptions() = default;
  SubcellOptions(double initial_data_rdmp_delta0,
                 double initial_data_rdmp_epsilon, double rdmp_delta0,
                 double rdmp_epsilon, double initial_data_persson_exponent,
                 double persson_exponent, bool always_use_subcells);

  void pup(PUP::er& p);

  double initial_data_rdmp_delta0() const { return initial_data_rdmp_delta0_; }

  double initial_data_rdmp_epsilon() const {
    return initial_data_rdmp_epsilon_;
  }

  double rdmp_delta0() const { return rdmp_delta0_; }

  double rdmp_epsilon() const { return rdmp_epsilon_; }

  double initial_data_persson_exponent() const {
    return initial_data_persson_exponent_;
  }

  double persson_exponent() const { return persson_exponent_; }

  bool always_use_subcells() const { return always_use_subcells_; }

 private:
  double initial_data_rdmp_delta0_ =
      std::numeric_limits<double>::signaling_NaN();
  double initial_data_rdmp_epsilon_ =
      std::numeric_limits<double>::signaling_NaN();
  double rdmp_delta0_ = std::numeric_limits<double>::signaling_NaN();
  double rdmp_epsilon_ = std::numeric_limits<double>::signaling_NaN();
  double initial_data_persson_exponent_ =
      std::numeric_limits<double>::signaling_NaN();
  double persson_exponent_ = std::numeric_limits<double>::signaling_NaN();
  bool always_use_subcells_ = false;
};

bool operator==(const SubcellOptions& lhs, const SubcellOptions& rhs);

bool operator!=(const SubcellOptions& lhs, const SubcellOptions& rhs);
}  // namespace evolution::dg::subcell
