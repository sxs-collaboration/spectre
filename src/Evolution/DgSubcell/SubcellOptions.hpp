// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t VolumeDim>
class DomainCreator;
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
  /// Parameters related to the troubled cell indicator (TCI) that determines
  /// when to switch between DG and FD.
  struct TroubledCellIndicator {
    static constexpr Options::String help =
        "Parameters related to the troubled cell indicator (TCI) that "
        "determines when to switch between DG and FD.";
  };

  struct PerssonTci {
    static constexpr Options::String help =
        "Parameters related to the Persson TCI";
    using group = TroubledCellIndicator;
  };
  /// The exponent \f$\alpha\f$ passed to the Persson troubled-cell indicator
  struct PerssonExponent {
    static std::string name() { return "Exponent"; }
    static constexpr Options::String help{
        "The exponent at which the error should decrease with (N+1-M)"};
    using type = double;
    static constexpr type lower_bound() { return 1.0; }
    static constexpr type upper_bound() { return 10.0; }
    using group = PerssonTci;
  };
  /// The number of highest modes the Persson troubled-cell indicator monitors
  struct PerssonNumHighestModes {
    static std::string name() { return "NumHighestModes"; }
    static constexpr Options::String help{
        "The number of highest modes M the Persson TCI monitors."};
    using type = size_t;
    static constexpr type lower_bound() { return 1_st; }
    static constexpr type upper_bound() { return 10_st; }
    using group = PerssonTci;
  };

  struct RdmpTci {
    static constexpr Options::String help =
        "Parameters related to the relaxed discrete maximum principle TCI";
    using group = TroubledCellIndicator;
  };
  /// The \f$\delta_0\f$ parameter in the relaxed discrete maximum principle
  /// troubled-cell indicator
  struct RdmpDelta0 {
    static std::string name() { return "Delta0"; }
    static constexpr Options::String help{"Absolute jump tolerance parameter."};
    using type = double;
    static type lower_bound() { return 0.0; }
    using group = RdmpTci;
  };
  /// The \f$\epsilon\f$ parameter in the relaxed discrete maximum principle
  /// troubled-cell indicator
  struct RdmpEpsilon {
    static std::string name() { return "Epsilon"; }
    static constexpr Options::String help{
        "The jump-dependent relaxation constant."};
    using type = double;
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 1.0; }
    using group = RdmpTci;
  };
  /// If true, then we always use the subcell method, not DG.
  struct AlwaysUseSubcells {
    static constexpr Options::String help{
        "If true, then always use the subcell method (e.g. finite-difference) "
        "instead of DG."};
    using type = bool;
    using group = TroubledCellIndicator;
  };
  /// Method to use for reconstructing the DG solution from the subcell
  /// solution.
  struct SubcellToDgReconstructionMethod {
    static constexpr Options::String help{
        "Method to use for reconstructing the DG solution from the subcell "
        "solution."};
    using type = fd::ReconstructionMethod;
  };
  /// \brief Use a width-one halo of FD elements around any troubled element.
  ///
  /// This provides a buffer of FD subcells so that as a discontinuity moves
  /// from one element to another we do not get any Gibbs phenomenon. In the
  /// case where we evolve the spacetime metric (e.g. GH+GRMHD) a halo region
  /// provides a buffer in case the stellar surface is near an element
  /// boundary. Since the GH variables are interpolated using high-order
  /// unlimited reconstruction, they can run into issues with Gibbs phenomenon.
  struct UseHalo {
    using type = bool;
    static constexpr Options::String help = {
        "Use a width-one halo of FD elements around any troubled element."
        "\n"
        "This provides a buffer of FD subcells so that as a discontinuity "
        "moves from one element to another we do not get any Gibbs "
        "phenomenon."};
    using group = TroubledCellIndicator;
  };

  /// \brief A list of block names on which to never do subcell.
  ///
  /// Set to `None` to allow subcell in all blocks.
  struct OnlyDgBlocksAndGroups {
    using type =
        Options::Auto<std::vector<std::string>, Options::AutoLabel::None>;
    static constexpr Options::String help = {
        "A list of block and group names on which to never do subcell.\n"
        "Set to 'None' to not restrict where FD can be used."};
    using group = TroubledCellIndicator;
  };

  /// \brief The order of the FD derivative used.
  ///
  /// Must be one of 2, 4, 6, 8, or 10. If `Auto` then the derivative order is
  /// determined for you, typically the next-lowest even order compared with
  /// the reconstruction scheme order. E.g. for a 5th-order reconstruction we
  /// would use 4th order derivatives.
  struct FiniteDifferenceDerivativeOrder {
    using type = ::fd::DerivativeOrder;
    static constexpr Options::String help = {
        "The finite difference derivative order to use. If computed from the "
        "reconstruction, then the reconstruction method must support returning "
        "its reconstruction order."};
  };

  struct FdToDgTci {
    static constexpr Options::String help =
        "Options related to how quickly we switch from FD to DG.";
    using group = TroubledCellIndicator;
  };
  /// The number of time steps taken between calls to the TCI to check if we
  /// can go back to the DG grid. A value of `1` means every time step, while
  /// `2` means every other time step.
  struct NumberOfStepsBetweenTciCalls {
    static constexpr Options::String help{
        "The number of time steps taken between calls to the TCI to check if "
        "we can go back to the DG grid. A value of `1` means every time step, "
        "while `2` means every other time step."};
    using type = size_t;
    static constexpr type lower_bound() { return 1; }
    using group = FdToDgTci;
  };
  /// The number of time steps/TCI calls after a switch from DG to FD before we
  /// allow switching back to DG.
  struct MinTciCallsAfterRollback {
    static constexpr Options::String help{
        "The number of time steps/TCI calls after a switch from DG to FD "
        "before we allow switching back to DG."};
    using type = size_t;
    static constexpr type lower_bound() { return 1; }
    using group = FdToDgTci;
  };
  /// The number of time steps/TCI calls that the TCI needs to have decided
  /// switching to DG is fine before we actually do the switch.
  struct MinimumClearTcis {
    static constexpr Options::String help{
        "The number of time steps/TCI calls that the TCI needs to have decided "
        "switching to DG is fine before we actually do the switch."};
    using type = size_t;
    static constexpr type lower_bound() { return 1; }
    using group = FdToDgTci;
  };

  using options = tmpl::list<
      PerssonExponent, PerssonNumHighestModes, RdmpDelta0, RdmpEpsilon,
      AlwaysUseSubcells, SubcellToDgReconstructionMethod, UseHalo,
      OnlyDgBlocksAndGroups, FiniteDifferenceDerivativeOrder,
      NumberOfStepsBetweenTciCalls, MinTciCallsAfterRollback, MinimumClearTcis>;

  static constexpr Options::String help{
      "System-agnostic options for the DG-subcell method."};

  SubcellOptions() = default;
  SubcellOptions(
      double persson_exponent, size_t persson_num_highest_modes,
      double rdmp_delta0, double rdmp_epsilon, bool always_use_subcells,
      fd::ReconstructionMethod recons_method, bool use_halo,
      std::optional<std::vector<std::string>> only_dg_block_and_group_names,
      ::fd::DerivativeOrder finite_difference_derivative_order,
      size_t number_of_steps_between_tci_calls,
      size_t min_tci_calls_after_rollback, size_t min_clear_tci_before_dg);

  /// \brief Given an existing SubcellOptions that was created from block and
  /// group names, create one that stores block IDs.
  ///
  /// The `DomainCreator` is used to convert block and group names into IDs
  /// and also to check that all listed block names and groups are in the
  /// domain.
  ///
  /// \note This is a workaround since our option parser does not allow us to
  /// retrieve options specified somewhere completely different in the input
  /// file.
  template <size_t Dim>
  SubcellOptions(const SubcellOptions& subcell_options_with_block_names,
                 const DomainCreator<Dim>& domain_creator);

  void pup(PUP::er& p);

  double persson_exponent() const { return persson_exponent_; }

  size_t persson_num_highest_modes() const {
    return persson_num_highest_modes_;
  }

  double rdmp_delta0() const { return rdmp_delta0_; }

  double rdmp_epsilon() const { return rdmp_epsilon_; }

  bool always_use_subcells() const { return always_use_subcells_; }

  fd::ReconstructionMethod reconstruction_method() const {
    return reconstruction_method_;
  }

  bool use_halo() const { return use_halo_; }

  const std::vector<size_t>& only_dg_block_ids() const {
    ASSERT(only_dg_block_ids_.has_value(),
           "The block IDs on which we are only allowed to do DG have not been "
           "set.");
    return only_dg_block_ids_.value();
  }

  ::fd::DerivativeOrder finite_difference_derivative_order() const {
    return finite_difference_derivative_order_;
  }

  /// The number of time steps between when we check the TCI to see if we can
  /// switch from FD back to DG.
  size_t number_of_steps_between_tci_calls() const {
    return number_of_steps_between_tci_calls_;
  }

  /// The number of time steps after a rollback before we check the TCI to see
  /// if we can switch from FD back to DG.
  size_t min_tci_calls_after_rollback() const {
    return min_tci_calls_after_rollback_;
  }

  /// The number of times the TCI must have flagged that we can switch back to
  /// DG before doing so.
  ///
  /// Note that if we only check the TCI every
  /// `number_of_steps_between_tci_calls()` then it takes
  /// `number_of_steps_between_tci_calls * min_clear_tci_before_dg` time steps
  /// before switching from FD back to DG.
  ///
  /// `0 means
  size_t min_clear_tci_before_dg() const { return min_clear_tci_before_dg_; }

 private:
  friend bool operator==(const SubcellOptions& lhs, const SubcellOptions& rhs);

  double persson_exponent_ = std::numeric_limits<double>::signaling_NaN();
  size_t persson_num_highest_modes_ =
      std::numeric_limits<size_t>::signaling_NaN();
  double rdmp_delta0_ = std::numeric_limits<double>::signaling_NaN();
  double rdmp_epsilon_ = std::numeric_limits<double>::signaling_NaN();
  bool always_use_subcells_ = false;
  fd::ReconstructionMethod reconstruction_method_ =
      fd::ReconstructionMethod::AllDimsAtOnce;
  bool use_halo_{false};
  std::optional<std::vector<std::string>> only_dg_block_and_group_names_{};
  std::optional<std::vector<size_t>> only_dg_block_ids_{};
  ::fd::DerivativeOrder finite_difference_derivative_order_{};
  size_t number_of_steps_between_tci_calls_{1};
  size_t min_tci_calls_after_rollback_{1};
  size_t min_clear_tci_before_dg_{0};
};

bool operator!=(const SubcellOptions& lhs, const SubcellOptions& rhs);
}  // namespace evolution::dg::subcell
