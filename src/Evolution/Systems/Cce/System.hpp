// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/*!
 * \ingroup EvolutionSystemsGroup
 * \brief The set of utilities for performing Cauchy characteristic evolution
 * and Cauchy characteristic matching.
 *
 * \details Cauchy characteristic evolution (CCE) is a secondary nonlinear GR
 * evolution system that covers the domain extending from a spherical boundary
 * away from the strong-field regime, and extending all the way to future null
 * infinity \f$\mathcal I^+\f$. The evolution system is governed by five
 * hypersurface equations that are integrated radially along future null slices,
 * and one evolution equation that governs the evolution of one hypersurface to
 * the next.
 *
 * The mathematics of CCE are intricate, and SpECTRE's version implements a
 * number of tricks and improvements that are not yet present in other contexts.
 * For introductions to CCE generally, see papers \cite Bishop1997ik,
 * \cite Bishop1998uk, and \cite Barkett2019uae. Here we do not present a full
 * description of all of the mathematics, but instead just provide a high-level
 * roadmap of the SpECTRE utilities and how they come together in the CCE
 * system. This is intended as a map for maintainers of the codebase.
 *
 * First, worldtube data from a completed or running Cauchy evolution of the
 * Einstein field equations (currently the only one implemented in SpECTRE is
 * Generalized Harmonic) must be translated to Bondi spin-weighted scalars at
 * the extraction sphere. Relevant utilities for this conversion are
 * `Cce::WorldtubeDataManager`, `Cce::ReducedWorldtubeDataManager`,
 * `Cce::create_bondi_boundary_data`. Relevant parts of the parallel
 * infrastructure are `Cce::H5WorldtubeBoundary`,
 * `Cce::Actions::BoundaryComputeAndSendToEvolution`,
 * `Cce::Actions::RequestBoundaryData`, and
 * `Cce::Actions::ReceiveWorldtubeData`.
 *
 * The first hypersurface must be initialized with some reasonable starting
 * value for the evolved Bondi quantity \f$J\f$. There isn't a universal perfect
 * prescription for this, as a complete description would require, like the
 * Cauchy initial data problem, knowledge of the system arbitrarily far in the
 * past. A utility for assigning the initial data is `Cce::InitializeJ`.
 *
 * SpECTRE CCE is currently unique in implementing an additional gauge transform
 * after the worldtube boundary data is derived. This is performed to obtain an
 * asymptotically well-behaved gauge that is guaranteed to avoid logarithmic
 * behavior that has plagued other CCE implementations, and so that the
 * asymptotic computations can be as simple, fast, and reliable as possible.
 * Relevant utilities for the gauge transformation are
 * `Cce::GaugeAdjustedBoundaryValue` (see template specializations),
 * `Cce::GaugeUpdateTimeDerivatives`, `Cce::GaugeUpdateAngularFromCartesian`,
 * `Cce::GaugeUpdateJacobianFromCoordinates`, `Cce::GaugeUpdateInterpolator`,
 * `Cce::GaugeUpdateOmega`, and `Cce::InitializeGauge`.
 *
 * Next, the CCE system must evaluate the hypersurface differential equations.
 * There are five, in sequence, deriving \f$\beta, Q, U, W,\f$ and \f$H\f$. For
 * each of the five radial differential equations, first the products and
 * derivatives on the right-hand side must be evaluated, then the full
 * right-hand side of the equation must be computed, and finally the radial
 * differential equation is integrated. The equations have a hierarchical
 * structure, so the result for \f$\beta\f$ feeds into the radial differential
 * equation for \f$Q\f$, and both feed into \f$U\f$, and so on.
 *
 * Relevant utilities for computing the inputs to the hypersurface equations are
 * `Cce::PrecomputeCceDependencies` (see template specializations),
 * `Cce::mutate_all_precompute_cce_dependencies`, `Cce::PreSwshDerivatives` (see
 * template specializations), `Cce::mutate_all_pre_swsh_derivatives_for_tag`,
 * and `Cce::mutate_all_swsh_derivatives_for_tag`. There are a number of
 * typelists in `IntegrandInputSteps.hpp` that determine the set of quantities
 * to be evaluated in each of the five hypersurface steps.
 * Once the hypersurface equation inputs are computed, then a hypersurface
 * equation right-hand side can be evaluated via `Cce::ComputeBondiIntegrand`
 * (see template specializations). Then, the hypersurface equation may be
 * integrated via `Cce::RadialIntegrateBondi` (see template specializations).
 *
 * Relevant parts of the parallel infrastructure for performing the hypersurface
 * steps are: `Cce::CharacteristicEvolution`,
 * `Cce::Actions::CalculateIntegrandInputsForTag`, and
 * `Cce::Actions::PrecomputeGlobalCceDependencies`. Note that most of the
 * algorithmic steps are laid out in order in the phase-dependent action list of
 * `Cce::CharacteristicEvolution`.
 *
 * The time integration for the hyperbolic part of the CCE equations is
 * performed via \f$\partial_u J = H\f$, where \f$\partial_u\f$ represents
 * differentiation with respect to retarded time at fixed numerical radius
 * \f$y\f$.
 *
 * At this point, all of the Bondi quantities on a given hypersurface have been
 * evaluated, and we wish to output the relevant waveform quantities at
 * \f$\mathcal I^+\f$. This acts much like an additional step in the
 * hypersurface sequence, with inputs that need to be calculated before the
 * quantities of interest can be evaluated. The action
 * `Cce::Actions::CalculateScriInputs` performs the sequence of steps to obtain
 * those inputs, and the utilities `Cce::CalculateScriPlusValue` (see template
 * specializations) can be used to evaluate the desired outputs at
 * \f$\mathcal I^+\f$.
 *
 * Unfortunately, those quantities at \f$\mathcal I^+\f$ are not yet an
 * appropriate waveform output, because the time coordinate with which they are
 * evaluated is the simulation time, not an asymptotically inertial time. So,
 * instead of directly writing the waveform outputs, we must put them in a queue
 * to be interpolated once enough data points have been accumulated to perform a
 * reliable interpolation at a consistent cut of \f$\mathcal I^+\f$ at constant
 * inertial time. Utilities for calculating and evolving the asymptotic inertial
 * time are `Cce::InitializeScriPlusValue` and `Cce::CalculateScriPlusValue`
 * using arguments involving `Cce::Tags::InertialRetardedTime`. A utility for
 * managing the interpolation is `Cce::ScriPlusInterpolationManager`, and
 * relevant parts of the parallel infrastructure for manipulating the data into
 * the interpolator and writing the results to disk are
 * `Cce::Actions::InsertInterpolationScriData` and
 * `Cce::Actions::ScriObserveInterpolated`.
 *
 */
namespace Cce {

struct System {
  using variables_tag = ::Tags::Variables<Tags::BondiJ>;
};
}
