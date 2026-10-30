// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Elasticity::BoundaryConditions {

/*!
 * \brief A laser beam normally incident to the surface, with a profile given
 * by an incoherent sum of Hermite-Gaussian modes
 *
 * This boundary condition represents a laser beam that exerts pressure normal
 * to the surface of a reflecting material. The pressure is an incoherent
 * (intensity) sum over Hermite-Gaussian modes \f$(n_k, m_k)\f$ with weights
 * \f$w_k\f$:
 *
 * \f{align}
 * n_i \left(-T^{ij}\right) = -p \, n^j, \quad
 * p = \sum_k w_k
 * \frac{H_{n_k}^2(u / r_0) H_{m_k}^2(v / r_0)}
 * {2^{n_k + m_k} n_k! m_k!} \frac{e^{-\frac{r^2}{r_0^2}}}{\pi r_0^2}
 * \f}
 *
 * where \f$T^{ij} = -Y^{ijkl} S_{kl}\f$ is the stress in the sign convention
 * of `Elasticity::FirstOrderSystem` (so \f$-T^{ij}\f$ is the conventional
 * stress and the condition is a compressive normal traction for
 * \f$p > 0\f$), \f$n_i\f$ is the unit normal pointing _out_ of the surface,
 * \f$r\f$ is the coordinate distance from the origin in the plane
 * perpendicular to \f$n_i\f$, \f$(u, v)\f$ are coordinates in that plane,
 * \f$H_n\f$ are the physicists' Hermite polynomials, and \f$r_0\f$ is the
 * "beam width" parameter. Each mode is normalized so that it exerts unit
 * total force, i.e. the total force is \f$\sum_k w_k\f$. Negative weights
 * subtract pressure, e.g. for differential measurements such as TEM02 -
 * TEM20. For the single mode \f$(0, 0)\f$ with unit weight this reduces to
 * the Gaussian profile \f$e^{-r^2/r_0^2} / (\pi r_0^2)\f$.
 *
 * The in-plane basis is \f$\hat{u} \propto \hat{t} - (\hat{t} \cdot n) n\f$
 * with \f$\hat{t} = \hat{x}\f$, or \f$\hat{t} = \hat{y}\f$ if
 * \f$|n_x| > 0.9\f$, and \f$\hat{v} = n \times \hat{u}\f$. For a surface
 * normal to the z-axis the index \f$n\f$ therefore refers to the x-direction
 * and \f$m\f$ to the y-direction. The surface must be flat (constant normal).
 *
 * The beam width must be positive and finite, and `Modes` must contain at
 * least one mode. Both are checked when the object is created, also from an
 * input file, and violations are reported as parse errors. A NaN width in an
 * input file is stopped earlier, by the option parser's lower-bound
 * comparison under the floating-point exception traps, so only direct
 * construction reaches the constructor's NaN check.
 *
 * \par Beam width convention
 * \f$r_0\f$ is the 1/e _intensity_ radius of the Gaussian envelope,
 * \f$I \propto e^{-r^2/r_0^2}\f$, so the TEM00 FWHM is
 * \f$2 \sqrt{\ln 2}\, r_0\f$. It is related to the 1/e^2 intensity radius
 * \f$\omega\f$ (the usual optics beam radius,
 * \f$I \propto e^{-2 r^2/\omega^2}\f$) by \f$\omega = \sqrt{2} r_0\f$. We
 * follow the convention of \cite Lovelace2007tn, \cite Lovelace2017xyf, and
 * \cite Vu2023thn in defining the beam width, and other publications may
 * include a factor of \f$\sqrt{2}\f$ in its definition.
 *
 * This boundary condition is used to simulate thermal noise induced in a mirror
 * by the laser, as detailed for instance in \cite Lovelace2007tn,
 * \cite Lovelace2017xyf, and \cite Vu2023thn.
 * See also `Elasticity::Solutions::HalfSpaceMirror` for
 * an analytic solution that involves this boundary condition (TEM00 only),
 * which uses the same beam width convention.
 */
class LaserBeam : public elliptic::BoundaryConditions::BoundaryCondition<3> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<3>;

 public:
  struct BeamWidth {
    using type = double;
    static constexpr Options::String help =
        "The width r_0 of the beam: the 1/e intensity radius of the Gaussian "
        "envelope, I ~ exp(-r^2/r_0^2), so the TEM00 FWHM = 2 * sqrt(ln 2) * "
        "r_0. NOT the 1/e^2 intensity radius w, which is sqrt(2) * r_0.";
    // Rejects negative values. Lower bounds are inclusive, so zero (which
    // would divide by zero in the .cpp file) and infinity are rejected by the
    // constructor.
    static type lower_bound() { return 0.0; }
  };

  struct ModeInfo {
    struct N {
      using type = size_t;
      static constexpr Options::String help = "The Hermite-Gaussian index n";
    };
    struct M {
      using type = size_t;
      static constexpr Options::String help = "The Hermite-Gaussian index m";
    };
    struct Weight {
      using type = double;
      static constexpr Options::String help =
          "The intensity multiplier (can be negative)";
    };

    using options = tmpl::list<N, M, Weight>;
    static constexpr Options::String help =
        "A single mode in the incoherent sum.";

    ModeInfo() = default;
    ModeInfo(size_t in_n, size_t in_m, double in_weight)
        : n(in_n), m(in_m), weight(in_weight) {}

    size_t n{0};
    size_t m{0};
    double weight{0.0};

    void pup(PUP::er& p) {
      p | n;
      p | m;
      p | weight;
    }
  };

  struct PolynomialModes {
    using type = std::vector<ModeInfo>;
    static constexpr Options::String help =
        "List of modes to sum (at least one). Each entry is a map with keys "
        "N, M and Weight, e.g. '- {N: 0, M: 2, Weight: 1.}'. Use a single "
        "entry {N: 0, M: 0, Weight: 1.} for a TEM00 beam.";
    static std::string name() { return "Modes"; }
  };

  static constexpr Options::String help =
      "A laser beam with arbitrary Hermite-Gaussian incoherent summation.";
  using options = tmpl::list<BeamWidth, PolynomialModes>;

  LaserBeam() = default;
  LaserBeam(const LaserBeam&) = default;
  LaserBeam& operator=(const LaserBeam&) = default;
  LaserBeam(LaserBeam&&) = default;
  LaserBeam& operator=(LaserBeam&&) = default;
  ~LaserBeam() = default;

  /// \cond
  explicit LaserBeam(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(LaserBeam);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<LaserBeam>(*this);
  }

  /// The beam width must be positive and finite and `modes` must be
  /// non-empty. Both are checked in all build types and reported through
  /// `context`.
  LaserBeam(double beam_width, const std::vector<ModeInfo>& modes,
            const Options::Context& context = {});

  double beam_width() const { return beam_width_; }

  const std::vector<ModeInfo>& modes() const { return modes_; }

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    return {3, elliptic::BoundaryConditionType::Neumann};
  }

  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>,
                 ::Tags::Normalized<
                     domain::Tags::UnnormalizedFaceNormal<3, Frame::Inertial>>>;
  using volume_tags = tmpl::list<>;

  void apply(gsl::not_null<tnsr::I<DataVector, 3>*> displacement,
             gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_minus_stress,
             const tnsr::iJ<DataVector, 3>& deriv_displacement,
             const tnsr::I<DataVector, 3>& x,
             const tnsr::i<DataVector, 3>& face_normal) const;

  using argument_tags_linearized = tmpl::list<>;
  using volume_tags_linearized = tmpl::list<>;

  static void apply_linearized(
      gsl::not_null<tnsr::I<DataVector, 3>*> displacement,
      gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_minus_stress,
      const tnsr::iJ<DataVector, 3>& deriv_displacement);

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    p | beam_width_;
    p | modes_;
  }

 private:
  double beam_width_{std::numeric_limits<double>::signaling_NaN()};
  std::vector<ModeInfo> modes_;
};

// Comparison operators defined in .cpp
bool operator==(const LaserBeam& lhs, const LaserBeam& rhs);
bool operator!=(const LaserBeam& lhs, const LaserBeam& rhs);
bool operator==(const LaserBeam::ModeInfo& lhs, const LaserBeam::ModeInfo& rhs);
bool operator!=(const LaserBeam::ModeInfo& lhs, const LaserBeam::ModeInfo& rhs);

}  // namespace Elasticity::BoundaryConditions
