// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Elasticity::BoundaryConditions {

/*!
 * \brief A laser beam with Gaussian profile normally incident to the surface
 *
 * This boundary condition represents a laser beam with Gaussian profile that
 * exerts pressure normal to the surface of a reflecting material. The pressure
 * we are considering here is
 *
 * \f{align}
 * n_i T^{ij} = -n^j \frac{e^{-\frac{r^2}{r_0^2}}}{\pi r_0^2}
 * \f}
 *
 * where \f$n_i\f$ is the unit normal pointing _out_ of the surface, \f$r\f$ is
 * the coordinate distance from the origin in the plane perpendicular to
 * \f$n_i\f$ and \f$r_0\f$ is the "beam width" parameter. The pressure profile
 * and the angle of incidence can be generalized in future work. Note that we
 * follow the convention of \cite Lovelace2007tn and \cite Lovelace2017xyf in
 * defining the beam width, and other publications may include a a factor of
 * \f$\sqrt{2}\f$ in its definition.
 *
 * This boundary condition is used to simulate thermal noise induced in a mirror
 * by the laser, as detailed for instance in \cite Lovelace2007tn and
 * \cite Lovelace2017xyf. See also `Elasticity::Solutions::HalfSpaceMirror` for
 * an analytic solution that involves this boundary condition.
 */
class LaserBeam : public elliptic::BoundaryConditions::BoundaryCondition<3> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<3>;

 public:
  struct BeamWidth {
    using type = double;
    static constexpr Options::String help =
        "The width r_0 of the Gaussian beam profile, such that FWHM = 2 * "
        "sqrt(ln 2) * r_0";
    static type lower_bound() { return 0.0; }
  };

  static constexpr Options::String help =
      "A laser beam with Gaussian profile normally incident to the surface.";
  using options = tmpl::list<BeamWidth>;

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

  LaserBeam(double beam_width) : beam_width_(beam_width) {}

  double beam_width() const { return beam_width_; }

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
             const tnsr::I<DataVector, 3>& x,
             const tnsr::i<DataVector, 3>& face_normal) const;

  using argument_tags_linearized = tmpl::list<>;
  using volume_tags_linearized = tmpl::list<>;

  static void apply_linearized(
      gsl::not_null<tnsr::I<DataVector, 3>*> displacement,
      gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_minus_stress);

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override { p | beam_width_; }

 private:
  double beam_width_{std::numeric_limits<double>::signaling_NaN()};
};

bool operator==(const LaserBeam& lhs, const LaserBeam& rhs);

bool operator!=(const LaserBeam& lhs, const LaserBeam& rhs);

}  // namespace Elasticity::BoundaryConditions
