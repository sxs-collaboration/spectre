// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <pup.h>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

namespace Xcts::BoundaryConditions {

/// How the shift of the negative-expansion solution enters the
/// `Xcts::BoundaryConditions::ApparentHorizon` boundary conditions
enum class NegativeExpansionShift : uint8_t {
  /// Impose the orthogonal part of the shift of the solution, as given
  None,
  /// Impose the orthogonal part of the shift of the solution, transformed to
  /// the comoving frame
  Orthogonal,
  /// Impose the full shift of the solution, transformed to the comoving frame
  AnalyticGuess
};

std::ostream& operator<<(std::ostream& os,
                         NegativeExpansionShift negative_expansion_shift);

/*!
 * \brief Impose the surface is a quasi-equilibrium apparent horizon.
 *
 * These boundary conditions on the conformal factor $\psi$, the lapse
 * $\alpha$ and the shift $\beta^i$ impose the surface is an apparent
 * horizon, i.e. that the expansion on the surface vanishes: $\Theta=0$.
 * Specifically, we impose:
 *
 * \begin{align}
 * \label{eq:ah_psi}
 * \bar{s}^k\bar{D}_k\psi &= -\frac{\psi^3}{8\alpha}\bar{s}_i\bar{s}_j\left(
 * (\bar{L}\beta)^{ij} - \bar{u}^{ij}\right)
 * - \frac{\psi}{4}\bar{m}^{ij}\bar{\nabla}_i\bar{s}_j + \frac{1}{6}K\psi^3
 * \\
 * \label{eq:ah_beta}
 * \beta_\mathrm{excess}^i &= \frac{\alpha}{\psi^2}\bar{s}^i
 * + \epsilon_{ijk}\Omega^j x^k - \beta_\mathrm{background}^i
 * \end{align}
 *
 * following section 7.2 of \cite Pfeiffer2005zm, section 12.3.2 of
 * \cite BaumgarteShapiro or section II.B.1 of \cite Varma2018sqd. In these
 * equations $\bar{s}_i$ is the conformal surface normal to the apparent
 * horizon, $\bar{m}^{ij}=\bar{\gamma}^{ij}-\bar{s}^i\bar{s}^j$ is the
 * induced conformal surface metric (denoted $\tilde{h}^{ij}$ in
 * \cite Pfeiffer2005zm and \cite Varma2018sqd) and $\bar{D}$ is the
 * covariant derivative w.r.t. to the conformal metric $\bar{\gamma}_{ij}$.
 * Note that e.g. in \cite Varma2018sqd Eq. (16) appears the surface-normal
 * $s^i$, not the _conformal_ surface normal $\bar{s}^i = \psi^2 s^i$.
 * To incur a spin on the apparent horizon we can freely choose the rotational
 * parameters $\boldsymbol{\Omega}$. Note that for a Kerr solution with
 * dimensionless spin $\boldsymbol{\chi}$ the rotational parameters at the
 * outer horizon are $\boldsymbol{\Omega} =
 * -\frac{\boldsymbol{\chi}}{2r_+}$, where $r_+ / M = 1 + \sqrt{1 -
 * \chi^2}$ (see e.g. Eq. (8) in \cite Ossokine2015yla). $\epsilon_{ijk}$
 * is the flat-space Levi-Civita symbol.
 *
 * Note that the quasi-equilibrium conditions don't restrict the boundary
 * condition for the lapse. The choice for the lapse boundary condition is made
 * by the `Lapse` input-file options. Currently implemented choices are:
 * - A zero von-Neumann boundary condition:
 *   $\bar{s}^k\bar{D}_k(\alpha\psi) = 0$
 * - A Dirichlet boundary condition imposing the lapse of an analytic solution.
 *
 * The `NegativeExpansion` option supports the following variants, explained
 * in the "Details" section below:
 * - Negative-expansion boundary conditions: place the excision surface inside
 *   the apparent horizon of a given Kerr solution.
 * - `ComovingShift: Orthogonal`: evaluate the Kerr shift in the comoving
 *   frame, for boosted black holes.
 * - `ComovingShift: AnalyticGuess`: impose the full comoving Kerr shift,
 *   including its tangential part.
 *
 * \note When negative-expansion boundary conditions are selected, the Kerr
 * solution gets evaluated at every boundary-condition application. This may
 * turn out to incur a significant computational cost, in which case a future
 * optimization might be to pre-compute the negative-expansion quantities at
 * initialization time and store them in the DataBox.
 *
 * ## Details
 *
 * ### Negative-expansion boundary conditions
 * This class also supports negative-expansion boundary conditions following
 * section II.B.2 of \cite Varma2018sqd by taking a Kerr solution as additional
 * parameter and computing its expansion at the excision surface. Choosing an
 * excision surface _within_ the apparent horizon of the Kerr solution will
 * result in a negative expansion that is added to the boundary condition for
 * the conformal factor. Therefore, the excision surface will lie within an
 * apparent horizon. Specifically, we add the quantity
 *
 * \begin{equation}
 * \frac{\psi^3}{4}\Theta_\mathrm{Kerr}
 * \end{equation}
 *
 * to ($\ref{eq:ah_psi}$), where $\Theta_\mathrm{Kerr}$ is the expansion
 * of the specified Kerr solution at the excision surface, and we add the
 * quantity
 *
 * \begin{equation}
 * \label{eq:ah_epsilon}
 * \epsilon = s_i \beta_\mathrm{Kerr}^i - \alpha_\mathrm{Kerr}
 * \end{equation}
 *
 * to the orthogonal part $s_i\beta_\mathrm{excess}^i$ of
 * ($\ref{eq:ah_beta}$).
 *
 * ### Comoving shift of the negative-expansion solution
 * A boosted Kerr solution is not stationary in the inertial frame: its
 * spatial metric moves with the black hole, so $\partial_t\gamma_{ij} \neq 0$
 * there. Its spatial metric is time independent only in the comoving frame,
 * i.e. after adding the background shift. The quasi-equilibrium conditions
 * assume exactly this, so the shift of the Kerr solution must be expressed in
 * the comoving frame for the negative-expansion corrections to be consistent.
 *
 * With `ComovingShift: Orthogonal` in the `NegativeExpansion` options, the
 * shift of the Kerr solution is transformed from the inertial frame to the
 * comoving frame in which the XCTS problem is posed,
 *
 * \begin{equation}
 * \beta_\mathrm{co}^i = \beta_\mathrm{Kerr}^i + \beta_\mathrm{background}^i,
 * \end{equation}
 *
 * evaluated at each point of the excision surface, and
 * $\beta_\mathrm{co}^i$ replaces $\beta_\mathrm{Kerr}^i$ in
 * ($\ref{eq:ah_epsilon}$). The Kerr solution must then be given in the
 * inertial frame, boosted with the background shift at its center so that it
 * is at rest in the comoving frame. For a single black hole this is its linear
 * velocity, and for a black hole centered at $c$ in a binary it is
 *
 * \begin{equation}
 * \Omega_0\times c + \dot{a}\,c + v_0.
 * \end{equation}
 *
 * The time derivatives of the background are assumed to vanish in the
 * comoving frame.
 *
 * ### Tangential correction from the analytic guess
 * With `ComovingShift: AnalyticGuess`, the shift of the Kerr solution is
 * transformed to the comoving frame as above, and also its tangential part,
 *
 * \begin{equation}
 * \tau^i = \beta_\mathrm{co}^i - \bar{s}^i\bar{s}_j\beta_\mathrm{co}^j,
 * \end{equation}
 *
 * is added to ($\ref{eq:ah_beta}$). Note that in this case $\epsilon$ uses
 * a _different normalization of the surface normal_: the unit normal of the
 * current spatial metric, $s_i = \psi^2\bar{s}_i$, instead of the unit normal
 * of the Kerr metric. Only with this normalization is the orthogonal part
 * $\bar{s}^i\bar{s}_j\beta_\mathrm{co}^j$ that $\epsilon$ contributes exactly
 * the part that $\tau^i$ removes, so that the orthogonal and tangential parts
 * recombine to the full comoving shift:
 *
 * \begin{equation}
 * \beta^i = \beta_\mathrm{co}^i
 * + \frac{\alpha - \alpha_\mathrm{Kerr}}{\psi^2}\bar{s}^i
 * + \epsilon_{ijk}\Omega^j x^k.
 * \end{equation}
 *
 * Then the boosted Kerr solution satisfies these boundary conditions on any
 * excision surface, not only on its apparent horizon, and for any conformal
 * background. The reason is that ($\ref{eq:ah_beta}$) fixes all components of
 * the shift on the surface. Without $\tau^i$ its tangential part is only the
 * rigid rotation $\epsilon_{ijk}\Omega^j x^k$, which in general differs from
 * the tangential part of the Kerr shift. With $\boldsymbol{\Omega}=0$ and
 * $\alpha=\alpha_\mathrm{Kerr}$, the imposed shift is $\beta_\mathrm{co}^i$ on
 * the whole surface, while the condition on $\psi$ is satisfied by the Kerr
 * data through the $\Theta_\mathrm{Kerr}$ term. Note that $\tau^i$ includes the
 * rotation of a spinning Kerr solution and the frame rotation of the background
 * shift around the black hole, so the `Rotation` parameters should typically be
 * set to zero in this case (in particular, without adding the orbital angular
 * velocity). The tangential correction may be useful for highly boosted,
 * spinning black holes: the frame dragging of a boosted Kerr solution, seen in
 * the comoving frame, is not a rigid rotation with the rest-frame value
 * $-\boldsymbol{\chi}/(2r_+)$, but it is imposed exactly through
 * $\tau^i$.
 */
template <Xcts::Geometry ConformalGeometry>
class ApparentHorizon
    : public elliptic::BoundaryConditions::BoundaryCondition<3> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<3>;

 public:
  static constexpr Options::String help =
      "Impose the boundary is a quasi-equilibrium apparent horizon.";

  struct Center {
    using type = std::array<double, 3>;
    static constexpr Options::String help =
        "The center of the excision surface representing the apparent-horizon "
        "surface";
  };
  struct Rotation {
    using type = std::array<double, 3>;
    static constexpr Options::String help =
        "The rotational parameters 'Omega' on the surface, which parametrize "
        "the spin of the black hole. The rotational parameters enter the "
        "Dirichlet boundary conditions for the shift in a term "
        "'Omega x (r - Center)', where 'r' are the coordinates on the surface.";
  };
  struct Lapse {
    using type =
        Options::Auto<std::unique_ptr<elliptic::analytic_data::InitialGuess>>;
    static constexpr Options::String help =
        "Specify an analytic solution or a superposed binary to impose a "
        "Dirichlet condition on the lapse. The analytic solution will be "
        "evaluated at coordinates centered at the apparent horizon. "
        "Alternatively, set this option to 'None' "
        "to impose a zero von-Neumann boundary condition on the lapse. Note "
        "that the latter will not result in the standard Kerr-Schild slicing "
        "for a single black hole.";
  };
  /// Options for negative-expansion boundary conditions
  struct NegativeExpansionOptions {
    struct Solution {
      using type = std::unique_ptr<elliptic::analytic_data::InitialGuess>;
      static constexpr Options::String help =
          "Analytic solution whose expansion is imposed at the excision "
          "surface. It is evaluated at coordinates centered at the apparent "
          "horizon.";
    };
    struct ComovingShift {
      using type = NegativeExpansionShift;
      static constexpr Options::String help =
          "How the shift of the solution is imposed at the excision surface. "
          "'None': impose the orthogonal part of the shift of the solution as "
          "given. 'Orthogonal': transform the shift of the solution from the "
          "inertial frame to the comoving frame, i.e. add the background shift "
          "to it, before imposing its orthogonal part. The solution must be "
          "boosted with the background shift at its center (e.g. the linear "
          "velocity of a single black hole), so that it is at rest in the "
          "comoving frame. 'AnalyticGuess': like 'Orthogonal', but "
          "impose the full comoving shift of the solution, including its "
          "tangential part. The tangential part includes the rotation of a "
          "spinning solution and the frame rotation of the background shift, "
          "so set 'Rotation' to zero (without the orbital angular velocity) "
          "unless an additional rotation is intended. May be useful for highly "
          "boosted, spinning black holes.";
    };
    using options = tmpl::list<Solution, ComovingShift>;
    static constexpr Options::String help =
        "Impose the expansion and shift of an analytic solution at the "
        "excision surface.";

    NegativeExpansionOptions() = default;
    NegativeExpansionOptions(
        std::unique_ptr<elliptic::analytic_data::InitialGuess> local_solution,
        NegativeExpansionShift local_shift)
        : solution(std::move(local_solution)), shift(local_shift) {}

    std::unique_ptr<elliptic::analytic_data::InitialGuess> solution{};
    NegativeExpansionShift shift{NegativeExpansionShift::None};
  };

  struct NegativeExpansion {
    using type =
        Options::Auto<NegativeExpansionOptions, Options::AutoLabel::None>;
    static constexpr Options::String help =
        "Specify an analytic solution to impose its expansion at the excision "
        "surface. The analytic solution will be evaluated at coordinates "
        "centered at the apparent horizon. "
        "If the excision surface lies within the solution's "
        "apparent horizon, the imposed expansion will be negative and thus the "
        "excision surface will lie within an apparent horizon. Alternatively, "
        "set this option to 'None' to impose the expansion is zero at the "
        "excision surface, meaning the excision surface _is_ an apparent "
        "horizon.";
  };

  using options = tmpl::list<Center, Rotation, Lapse, NegativeExpansion>;

  ApparentHorizon() = default;
  ApparentHorizon(const ApparentHorizon&) = delete;
  ApparentHorizon& operator=(const ApparentHorizon&) = delete;
  ApparentHorizon(ApparentHorizon&&) = default;
  ApparentHorizon& operator=(ApparentHorizon&&) = default;
  ~ApparentHorizon() = default;

  /// \cond
  explicit ApparentHorizon(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ApparentHorizon);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<ApparentHorizon>(
        center_, rotation_,
        solution_for_lapse_.has_value()
            ? std::make_optional(
                  serialize_and_deserialize<
                      std::unique_ptr<elliptic::analytic_data::InitialGuess>>(
                      solution_for_lapse_.value()))
            : std::nullopt,
        solution_for_negative_expansion_.has_value()
            ? std::make_optional(NegativeExpansionOptions{
                  serialize_and_deserialize<
                      std::unique_ptr<elliptic::analytic_data::InitialGuess>>(
                      solution_for_negative_expansion_.value()),
                  negative_expansion_shift_})
            : std::nullopt);
  }

  ApparentHorizon(
      std::array<double, 3> center, std::array<double, 3> rotation,
      std::optional<std::unique_ptr<elliptic::analytic_data::InitialGuess>>
          solution_for_lapse,
      std::optional<NegativeExpansionOptions> negative_expansion,
      const Options::Context& context = {});

  const std::array<double, 3>& center() const { return center_; }
  const std::array<double, 3>& rotation() const { return rotation_; }
  const std::optional<std::unique_ptr<elliptic::analytic_data::InitialGuess>>&
  solution_for_lapse() const {
    return solution_for_lapse_;
  }
  const std::optional<std::unique_ptr<elliptic::analytic_data::InitialGuess>>&
  solution_for_negative_expansion() const {
    return solution_for_negative_expansion_;
  }
  NegativeExpansionShift negative_expansion_shift() const {
    return negative_expansion_shift_;
  }

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    return {// Conformal factor
            elliptic::BoundaryConditionType::Neumann,
            // Lapse times conformal factor
            this->solution_for_lapse_.has_value()
                ? elliptic::BoundaryConditionType::Dirichlet
                : elliptic::BoundaryConditionType::Neumann,
            // Shift
            elliptic::BoundaryConditionType::Dirichlet,
            elliptic::BoundaryConditionType::Dirichlet,
            elliptic::BoundaryConditionType::Dirichlet};
  }

  using argument_tags = tmpl::flatten<tmpl::list<
      domain::Tags::FaceNormal<3>,
      ::Tags::deriv<domain::Tags::UnnormalizedFaceNormal<3>, tmpl::size_t<3>,
                    Frame::Inertial>,
      domain::Tags::UnnormalizedFaceNormalMagnitude<3>,
      domain::Tags::Coordinates<3, Frame::Inertial>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      Tags::ShiftBackground<DataVector, 3, Frame::Inertial>,
      Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<DataVector, 3,
                                                              Frame::Inertial>,
      tmpl::conditional_t<ConformalGeometry == Xcts::Geometry::Curved,
                          tmpl::list<Tags::InverseConformalMetric<
                                         DataVector, 3, Frame::Inertial>,
                                     Tags::ConformalChristoffelSecondKind<
                                         DataVector, 3, Frame::Inertial>>,
                          tmpl::list<>>>>;
  using volume_tags = tmpl::list<>;

  void apply(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
      gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor_minus_one,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
      gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient,
      gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_longitudinal_shift_excess,
      const tnsr::i<DataVector, 3>& deriv_conformal_factor,
      const tnsr::i<DataVector, 3>& deriv_lapse_times_conformal_factor,
      const tnsr::iJ<DataVector, 3>& deriv_shift_excess,
      const tnsr::i<DataVector, 3>& face_normal,
      const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
      const Scalar<DataVector>& face_normal_magnitude,
      const tnsr::I<DataVector, 3>& x,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const tnsr::I<DataVector, 3>& shift_background,
      const tnsr::II<DataVector, 3>& longitudinal_shift_background) const;

  void apply(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
      gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor_minus_one,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
      gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient,
      gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_longitudinal_shift_excess,
      const tnsr::i<DataVector, 3>& deriv_conformal_factor,
      const tnsr::i<DataVector, 3>& deriv_lapse_times_conformal_factor,
      const tnsr::iJ<DataVector, 3>& deriv_shift_excess,
      const tnsr::i<DataVector, 3>& face_normal,
      const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
      const Scalar<DataVector>& face_normal_magnitude,
      const tnsr::I<DataVector, 3>& x,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const tnsr::I<DataVector, 3>& shift_background,
      const tnsr::II<DataVector, 3>& longitudinal_shift_background,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) const;

  using argument_tags_linearized = tmpl::flatten<tmpl::list<
      ::Tags::Normalized<
          domain::Tags::UnnormalizedFaceNormal<3, Frame::Inertial>>,
      ::Tags::deriv<domain::Tags::UnnormalizedFaceNormal<3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>,
      ::Tags::Magnitude<
          domain::Tags::UnnormalizedFaceNormal<3, Frame::Inertial>>,
      domain::Tags::Coordinates<3, Frame::Inertial>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      Tags::ShiftBackground<DataVector, 3, Frame::Inertial>,
      Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<DataVector, 3,
                                                              Frame::Inertial>,
      Tags::ConformalFactorMinusOne<DataVector>,
      Tags::LapseTimesConformalFactorMinusOne<DataVector>,
      ::Tags::NormalDotFlux<Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>,
      tmpl::conditional_t<ConformalGeometry == Xcts::Geometry::Curved,
                          tmpl::list<Tags::InverseConformalMetric<
                                         DataVector, 3, Frame::Inertial>,
                                     Tags::ConformalChristoffelSecondKind<
                                         DataVector, 3, Frame::Inertial>>,
                          tmpl::list<>>>>;
  using volume_tags_linearized = tmpl::list<>;

  void apply_linearized(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
      gsl::not_null<Scalar<DataVector>*>
          lapse_times_conformal_factor_correction,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_conformal_factor_gradient_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient_correction,
      gsl::not_null<tnsr::I<DataVector, 3>*>
          n_dot_longitudinal_shift_excess_correction,
      const tnsr::i<DataVector, 3>& deriv_conformal_factor_correction,
      const tnsr::i<DataVector, 3>&
          deriv_lapse_times_conformal_factor_correction,
      const tnsr::iJ<DataVector, 3>& deriv_shift_excess_correction,
      const tnsr::i<DataVector, 3>& face_normal,
      const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
      const Scalar<DataVector>& face_normal_magnitude,
      const tnsr::I<DataVector, 3>& x,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const tnsr::I<DataVector, 3>& shift_background,
      const tnsr::II<DataVector, 3>& longitudinal_shift_background,
      const Scalar<DataVector>& conformal_factor_minus_one,
      const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
      const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess) const;

  void apply_linearized(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
      gsl::not_null<Scalar<DataVector>*>
          lapse_times_conformal_factor_correction,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_conformal_factor_gradient_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient_correction,
      gsl::not_null<tnsr::I<DataVector, 3>*>
          n_dot_longitudinal_shift_excess_correction,
      const tnsr::i<DataVector, 3>& deriv_conformal_factor_correction,
      const tnsr::i<DataVector, 3>&
          deriv_lapse_times_conformal_factor_correction,
      const tnsr::iJ<DataVector, 3>& deriv_shift_excess_correction,
      const tnsr::i<DataVector, 3>& face_normal,
      const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
      const Scalar<DataVector>& face_normal_magnitude,
      const tnsr::I<DataVector, 3>& x,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const tnsr::I<DataVector, 3>& shift_background,
      const tnsr::II<DataVector, 3>& longitudinal_shift_background,
      const Scalar<DataVector>& conformal_factor_minus_one,
      const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
      const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  std::array<double, 3> center_ =
      make_array<3>(std::numeric_limits<double>::signaling_NaN());
  std::array<double, 3> rotation_ =
      make_array<3>(std::numeric_limits<double>::signaling_NaN());
  std::optional<std::unique_ptr<elliptic::analytic_data::InitialGuess>>
      solution_for_lapse_{};
  std::optional<std::unique_ptr<elliptic::analytic_data::InitialGuess>>
      solution_for_negative_expansion_{};
  NegativeExpansionShift negative_expansion_shift_{
      NegativeExpansionShift::None};
};

}  // namespace Xcts::BoundaryConditions

/// \cond
template <>
struct Options::create_from_yaml<
    Xcts::BoundaryConditions::NegativeExpansionShift> {
  template <typename Metavariables>
  static Xcts::BoundaryConditions::NegativeExpansionShift create(
      const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
Xcts::BoundaryConditions::NegativeExpansionShift
Options::create_from_yaml<Xcts::BoundaryConditions::NegativeExpansionShift>::
    create<void>(const Options::Option& options);
/// \endcond
