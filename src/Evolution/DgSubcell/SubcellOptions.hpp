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

  using options =
      tmpl::list<InitialDataRdmpDelta0, InitialDataRdmpEpsilon, RdmpDelta0,
                 RdmpEpsilon, InitialDataPerssonExponent, PerssonExponent,
                 AlwaysUseSubcells, SubcellToDgReconstructionMethod, UseHalo,
                 OnlyDgBlocksAndGroups, FiniteDifferenceDerivativeOrder>;

  static constexpr Options::String help{
      "System-agnostic options for the DG-subcell method."};

  SubcellOptions() = default;
  SubcellOptions(
      double initial_data_rdmp_delta0, double initial_data_rdmp_epsilon,
      double rdmp_delta0, double rdmp_epsilon,
      double initial_data_persson_exponent, double persson_exponent,
      bool always_use_subcells, fd::ReconstructionMethod recons_method,
      bool use_halo,
      std::optional<std::vector<std::string>> only_dg_block_and_group_names,
      ::fd::DerivativeOrder finite_difference_derivative_order);

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

 private:
  friend bool operator==(const SubcellOptions& lhs, const SubcellOptions& rhs);

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
  fd::ReconstructionMethod reconstruction_method_ =
      fd::ReconstructionMethod::AllDimsAtOnce;
  bool use_halo_{false};
  std::optional<std::vector<std::string>> only_dg_block_and_group_names_{};
  std::optional<std::vector<size_t>> only_dg_block_ids_{};
  ::fd::DerivativeOrder finite_difference_derivative_order_{};
};

bool operator!=(const SubcellOptions& lhs, const SubcellOptions& rhs);
}  // namespace evolution::dg::subcell
