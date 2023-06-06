// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Options/Options.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class ComplexDataVector;
/// \endcond

namespace Cce {
namespace InitializeJ {

/// Possible iteration heuristics to use for optimizing the value of the
/// conformal factor \f$\omega\f$ to fix the initial data.
enum class ConformalFactorIterationHeuristic {
  /// Assumes that the spin-weighted Jacobian perturbations obey
  /// \f$c = \hat \eth f\f$,\f$d = \hat{\bar\eth} f\f$, for some spin-weight-1
  /// value\f$f\f$.
  SpinWeight1CoordPerturbation,
  /// Varies only the \f$d\f$ spin-weighted Jacobian when constructing the
  /// itertion heuristic, leaving \f$c\f$ fixed
  OnlyVaryGaugeD
};

/*!
 * \brief Generate initial data that has a conformal factor \f$\omega\f$ chosen
 * to compensate for the boundary value of \f$\beta\f$ so that the initial time
 * coordinate is approximately inertial at \f$I^+\f$.
 *
 * \details The core calculation for this initial data choice is the iterative
 * optimization of the angular conformal factor \f$\omega\f$ such that it
 * cancels some portion of the value of \f$e^{2\beta}\f$ that contributes to the
 * definition of the asymptotically inertial time.
 * The initial data generation process proceeds slightly differently depending
 * on the set of input options that are used:
 * - If `UseBetaIntegralEstimate` is false, the conformal factor will be
 * optimized to minimize the transformed value of \f$\beta\f$ on the worldtube
 * boundary.
 * - If `UseBetaIntegralEstimate` is true, the conformal factor will be
 * optimized to minimize an estimate of the asymptotic value of \f$\beta\f$ in
 * the evolved coordinates.
 * - `OptimizeL0Mode` indicates whether the \f$l=0\f$ mode of the conformal
 * factor shoudl be included in the optimization. This option is useful because
 * the optimization can usually find a better solution when the \f$l=0\f$ mode
 * is ignored, and the \f$l=0\f$ should not contribute significantly to the
 * resulting waveform.
 *
 * - If `UseInputModes` is false, the \f$J\f$ value on the initial hypersurface
 * will be set by an \f$A/r + B/r^3\f$ ansatz, chosen to match the worldtube
 * boundary value of \f$J\f$ and \f$\partial_r J\f$ in the new coordinates.
 * In this case, the alternative arguments `InputModes` or `InputModesFromFile`
 * are ignored.
 * - If `UseInputModes` is true, the \f$1/r\f$ part of \f$J\f$ will be set to
 * spin-weighted spherical harmonic modes specified by either an input h5 file
 * (in the case of using the input option `InputModesFromFile`) or from a list
 * of complex values specified in the input file (in the case of using the input
 * option `InputModes`). Then, the \f$1/r^3\f$ and \f$1/r^4\f$ parts of \f$J\f$
 * are chosen to match the boundary value of \f$J\f$ and \f$\partial_r J\f$ on
 * the worldtube boundary in the new coordinates.
 */
struct ConformalFactor : InitializeJ<false> {
  struct AngularCoordinateTolerance {
    using type = double;
    static std::string name() { return "AngularCoordTolerance"; }
    static constexpr Options::String help = {
        "Tolerance of initial angular coordinates for CCE"};
    static type lower_bound() { return 1.0e-14; }
    static type upper_bound() { return 1.0e-3; }
  };
  struct MaxIterations {
    using type = size_t;
    static constexpr Options::String help = {
        "Number of linearized inversion iterations."};
    static type lower_bound() { return 10; }
    static type upper_bound() { return 1000; }
    static type suggested_value() { return 300; }
  };
  struct RequireConvergence {
    using type = bool;
    static constexpr Options::String help = {
        "If true, initialization will error if it hits MaxIterations"};
    static type suggested_value() { return true; }
  };
  struct OptimizeL0Mode {
    using type = bool;
    static constexpr Options::String help = {
        "If true, the average value of the conformal factor will be included "
        "during optimization; otherwise it will be omitted (filtered)."};
    static type suggested_value() { return false; }
  };
  struct UseBetaIntegralEstimate {
    using type = bool;
    static constexpr Options::String help = {
        "If true, the iterative algorithm will calculate an estimate of the "
        "asymptotic beta value using the 1/r part of the initial J."};
    static type suggested_value() { return true; }
  };
  struct ConformalFactorIterationHeuristic {
    using type = ::Cce::InitializeJ::ConformalFactorIterationHeuristic;
    static constexpr Options::String help = {
        "The heuristic method used to set the spin-weighted Jacobian factors "
        "when iterating to minimize the asymptotic conformal factor."};
    static type suggested_value() {
      return ::Cce::InitializeJ::ConformalFactorIterationHeuristic::
          SpinWeight1CoordPerturbation;
    }
  };
  struct UseInputModes {
    using type = bool;
    static constexpr Options::String help = {
        "If true, the 1/r part of J will be set using modes read from the "
        "input file, or from a specified h5 file. If false, the inverse cubic "
        "scheme will determine the 1/r part of J."};
  };
  struct InputModesFromFile {
    using type = std::string;
    static constexpr Options::String help = {
        "A filename from which to retrieve a set of modes (from InitialJ.dat) "
        "to use to determine the 1/r part of J on the initial hypersurface. "
        "The modes are parsed in l-ascending, m-ascending, m-varies-fastest, "
        "real then imaginary part order."};
  };
  struct InputModes {
    using type = std::vector<std::complex<double>>;
    static constexpr Options::String help = {
        "An explicit list of modes to use to set the 1/r part of J on the "
        "initial hypersurface. They are parsed in l-ascending, m-ascending, "
        "m-varies-fastest order."};
  };

  using options =
      tmpl::list<AngularCoordinateTolerance, MaxIterations, RequireConvergence,
                 OptimizeL0Mode, UseBetaIntegralEstimate,
                 ConformalFactorIterationHeuristic, UseInputModes,
                 Options::Alternatives<tmpl::list<InputModesFromFile>,
                                       tmpl::list<InputModes>>>;
  static constexpr Options::String help = {
      "Generate CCE initial data based on choosing an angular conformal factor "
      "based on the value of the CCE scalar beta in an attempt to make the "
      "time variable approximately asymptotically inertial"};

  WRAPPED_PUPable_decl_template(ConformalFactor);  // NOLINT
  explicit ConformalFactor(CkMigrateMessage* msg);

  ConformalFactor() = default;
  ConformalFactor(
      double angular_coordinate_tolerance, size_t max_iterations,
      bool require_convergence, bool optimize_l_0_mode,
      bool use_beta_integral_estimate,
      ::Cce::InitializeJ::ConformalFactorIterationHeuristic iteration_heuristic,
      bool use_input_modes, std::string input_mode_filename);

  ConformalFactor(
      double angular_coordinate_tolerance, size_t max_iterations,
      bool require_convergence, bool optimize_l_0_mode,
      bool use_beta_integral_estimate,
      ::Cce::InitializeJ::ConformalFactorIterationHeuristic iteration_heuristic,
      bool use_input_modes, std::vector<std::complex<double>> input_modes);

  std::unique_ptr<InitializeJ> get_clone() const override;

  void operator()(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
      gsl::not_null<
          tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
          angular_cauchy_coordinates,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta, size_t l_max,
      size_t number_of_radial_points,
      gsl::not_null<Parallel::NodeLock*> hdf5_lock) const override;

  void pup(PUP::er& p) override;

 private:
  double angular_coordinate_tolerance_ = 1.0e-11;
  size_t max_iterations_ = 300;
  bool require_convergence_ = true;
  bool optimize_l_0_mode_ = false;
  bool use_beta_integral_estimate_ = true;
  ::Cce::InitializeJ::ConformalFactorIterationHeuristic iteration_heuristic_ =
      ::Cce::InitializeJ::ConformalFactorIterationHeuristic::
          SpinWeight1CoordPerturbation;
  bool use_input_modes_ = false;
  std::vector<std::complex<double>> input_modes_;
  std::optional<std::string> input_mode_filename_;
};

std::ostream& operator<<(
    std::ostream& os,
    const Cce::InitializeJ::ConformalFactorIterationHeuristic& heuristic_type);

}  // namespace InitializeJ
}  // namespace Cce

template <>
struct Options::create_from_yaml<
    Cce::InitializeJ::ConformalFactorIterationHeuristic> {
  template <typename Metavariables>
  static Cce::InitializeJ::ConformalFactorIterationHeuristic create(
      const Options::Option& options) {
    return create<void>(options);
  }
};
template <>
Cce::InitializeJ::ConformalFactorIterationHeuristic
Options::create_from_yaml<Cce::InitializeJ::ConformalFactorIterationHeuristic>::
    create<void>(const Options::Option& options);
