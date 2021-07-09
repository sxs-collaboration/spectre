// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Xcts::BoundaryConditions {
namespace detail {

template <Xcts::Geometry ConformalGeometry>
struct ApparentHorizonImpl {
  static constexpr Options::String help =
      "Impose the boundary is a quasi-equilibrium apparent horizon.";

  struct Center {
    using type = std::array<double, 3>;
    static constexpr Options::String help =
        "The center of the coordinate sphere representing the apparent-horizon "
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
    using type = Options::Auto<gr::Solutions::KerrSchild>;
    static constexpr Options::String help =
        "Specify a Kerr solution to impose a Dirichlet condition on the lapse "
        "in Kerr-Schild coordinates. Alternatively, set this option to 'None' "
        "to impose a zero von-Neumann boundary condition on the lapse. Note "
        "that the latter will not result in the standard Kerr-Schild slicing "
        "for a single black hole.";
  };

  using options = tmpl::list<Center, Rotation, Lapse>;

  ApparentHorizonImpl() = default;
  ApparentHorizonImpl(const ApparentHorizonImpl&) = default;
  ApparentHorizonImpl& operator=(const ApparentHorizonImpl&) = default;
  ApparentHorizonImpl(ApparentHorizonImpl&&) = default;
  ApparentHorizonImpl& operator=(ApparentHorizonImpl&&) = default;
  ~ApparentHorizonImpl() = default;

  ApparentHorizonImpl(std::array<double, 3> center,
                      std::array<double, 3> rotation,
                      std::optional<gr::Solutions::KerrSchild>
                          kerr_solution_for_lapse) noexcept;

  const std::array<double, 3>& center() const noexcept { return center_; }
  const std::array<double, 3>& rotation() const noexcept { return rotation_; }
  const std::optional<gr::Solutions::KerrSchild>& kerr_solution_for_lapse()
      const noexcept {
    return kerr_solution_for_lapse_;
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
      gsl::not_null<Scalar<DataVector>*> conformal_factor,
      gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
      gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient,
      gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_longitudinal_shift_excess,
      const tnsr::i<DataVector, 3>& face_normal,
      const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
      const Scalar<DataVector>& face_normal_magnitude,
      const tnsr::I<DataVector, 3>& x,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const tnsr::I<DataVector, 3>& shift_background,
      const tnsr::II<DataVector, 3>& longitudinal_shift_background)
      const noexcept;

  void apply(
      gsl::not_null<Scalar<DataVector>*> conformal_factor,
      gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
      gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient,
      gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_longitudinal_shift_excess,
      const tnsr::i<DataVector, 3>& face_normal,
      const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
      const Scalar<DataVector>& face_normal_magnitude,
      const tnsr::I<DataVector, 3>& x,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const tnsr::I<DataVector, 3>& shift_background,
      const tnsr::II<DataVector, 3>& longitudinal_shift_background,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind)
      const noexcept;

  using argument_tags_linearized = tmpl::flatten<tmpl::list<
      ::Tags::Normalized<
          domain::Tags::UnnormalizedFaceNormal<3, Frame::Inertial>>,
      ::Tags::deriv<domain::Tags::UnnormalizedFaceNormal<3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>,
      ::Tags::Magnitude<
          domain::Tags::UnnormalizedFaceNormal<3, Frame::Inertial>>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<DataVector, 3,
                                                              Frame::Inertial>,
      Tags::ConformalFactor<DataVector>,
      Tags::LapseTimesConformalFactor<DataVector>,
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
      const tnsr::i<DataVector, 3>& face_normal,
      const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
      const Scalar<DataVector>& face_normal_magnitude,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const tnsr::II<DataVector, 3>& longitudinal_shift_background,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess)
      const noexcept;

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
      const tnsr::i<DataVector, 3>& face_normal,
      const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
      const Scalar<DataVector>& face_normal_magnitude,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const tnsr::II<DataVector, 3>& longitudinal_shift_background,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind)
      const noexcept;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

 private:
  std::array<double, 3> center_ =
      make_array<3>(std::numeric_limits<double>::signaling_NaN());
  std::array<double, 3> rotation_ =
      make_array<3>(std::numeric_limits<double>::signaling_NaN());
  std::optional<gr::Solutions::KerrSchild> kerr_solution_for_lapse_{};
};

template <Xcts::Geometry ConformalGeometry>
bool operator==(const ApparentHorizonImpl<ConformalGeometry>& lhs,
                const ApparentHorizonImpl<ConformalGeometry>& rhs) noexcept;

template <Xcts::Geometry ConformalGeometry>
bool operator!=(const ApparentHorizonImpl<ConformalGeometry>& lhs,
                const ApparentHorizonImpl<ConformalGeometry>& rhs) noexcept;

}  // namespace detail

// The following implements the registration and factory-creation mechanism

/// \cond
template <Xcts::Geometry ConformalGeometry, typename Registrars>
struct ApparentHorizon;

namespace Registrars {
template <Xcts::Geometry ConformalGeometry>
struct ApparentHorizon {
  template <typename Registrars>
  using f = BoundaryConditions::ApparentHorizon<ConformalGeometry, Registrars>;
};
}  // namespace Registrars
/// \endcond

/*!
 * \brief Impose the surface is a quasi-equilibrium apparent horizon.
 *
 * These boundary conditions on the conformal factor \f$\psi\f$, the lapse
 * \f$\alpha\f$ and the shift \f$\beta^i\f$ impose the surface is an apparent
 * horizon, i.e. that the expansion on the surface vanishes: \f$\Theta=0\f$.
 * Specifically, we impose:
 *
 * \f{align}
 * \bar{s}^k\bar{D}_k\psi &= -\frac{\psi^3}{8\alpha}\bar{s}_i\bar{s}_j\left(
 * (\bar{L}\beta)^{ij} - \bar{u}^{ij}\right)
 * - \frac{\psi}{4}\bar{m}^{ij}\bar{\nabla}_i\bar{s}_j + \frac{1}{6}K\psi^3
 * \\
 * \beta_\mathrm{excess}^i &= \frac{\alpha}{\psi^2}\bar{s}^i
 * + \epsilon_{ijk}\Omega^j x^k
 * \f}
 *
 * following section 7.2 of \cite Pfeiffer2005zm, section 12.3.2 of
 * \cite BaumgarteShapiro or section B.1 of \cite Varma2018sqd. In these
 * equations \f$\bar{s}_i\f$ is the conformal surface normal to the apparent
 * horizon, \f$\bar{m}^ij=\bar{\gamma}^{ij}-\bar{s}^i\bar{s}^j\f$ is the induced
 * conformal surface metric (denoted \f$\tilde{h}^{ij}\f$ in
 * \cite Pfeiffer2005zm and \cite Varma2018sqd) and \f$\bar{D}\f$ is the
 * covariant derivative w.r.t. to the conformal metric \f$\bar{\gamma}_{ij}\f$.
 * To incur a spin on the apparent horizon we can freely choose the rotational
 * parameters \f$\boldsymbol{\Omega}\f$. Note that for a Kerr solution
 * with dimensionless spin \f$\boldsymbol{\chi}\f$ the rotational parameters at
 * the outer horizon are \f$\boldsymbol{Omega} =
 * -\frac{\boldsymbol{\chi}}{2r_+}\f$, where \f$r_+ / M = 1 + \sqrt{1 -
 * \chi^2}\f$ (see e.g. Eq. (8) in \cite Ossokine2015yla).
 *
 * Note that the quasi-equilibrium conditions don't restrict the boundary
 * condition for the lapse. Currently implemented choices are:
 * - A zero von-Neumann boundary condition:
 *   \f$\bar{s}^k\bar{D}_k(\alpha\psi) = 0\f$
 * - A Dirichlet boundary condition imposing the lapse of a Kerr-Schild analytic
 *   solution.
 *
 * \par Negative-expansion boundary conditions:
 * Support for negative-expansion boundary conditions following section B.2 of
 * \cite Varma2018sqd can be added here by taking a Kerr solution as additional
 * parameter and computing its expansion at the excision surface. Choosing an
 * excision surface _within_ the apparent horizon of the Kerr solution will
 * result in a negative expansion that can be added to the boundary condition
 * for the conformal factor.
 */
template <Xcts::Geometry ConformalGeometry,
          typename Registrars =
              tmpl::list<Registrars::ApparentHorizon<ConformalGeometry>>>
class ApparentHorizon
    : public elliptic::BoundaryConditions::BoundaryCondition<3, Registrars>,
      public detail::ApparentHorizonImpl<ConformalGeometry> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<3, Registrars>;

 public:
  ApparentHorizon() = default;
  ApparentHorizon(const ApparentHorizon&) = default;
  ApparentHorizon& operator=(const ApparentHorizon&) = default;
  ApparentHorizon(ApparentHorizon&&) = default;
  ApparentHorizon& operator=(ApparentHorizon&&) = default;
  ~ApparentHorizon() = default;

  using detail::ApparentHorizonImpl<ConformalGeometry>::ApparentHorizonImpl;

  /// \cond
  explicit ApparentHorizon(CkMigrateMessage* m) noexcept : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ApparentHorizon);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const noexcept override {
    return std::make_unique<ApparentHorizon>(*this);
  }

  void pup(PUP::er& p) noexcept override {
    Base::pup(p);
    detail::ApparentHorizonImpl<ConformalGeometry>::pup(p);
  }
};

/// \cond
template <Xcts::Geometry ConformalGeometry, typename Registrars>
PUP::able::PUP_ID ApparentHorizon<ConformalGeometry, Registrars>::my_PUP_ID =
    0;  // NOLINT
/// \endcond

}  // namespace Xcts::BoundaryConditions
