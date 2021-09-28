// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <pup.h>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/AnalyticSolution.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Elasticity::Solutions {

namespace detail {
template <typename DataType>
struct BentBeamVariables {
  using Cache = CachedTempBuffer<BentBeamVariables, Tags::Displacement<2>,
                                 Tags::Strain<2>, Tags::MinusStress<2>,
                                 Tags::PotentialEnergyDensity<2>,
                                 ::Tags::FixedSource<Tags::Displacement<2>>>;

  const tnsr::I<DataType, 2>& x;
  const double length;
  const double height;
  const double bending_moment;
  const ConstitutiveRelations::IsotropicHomogeneous<2>& constitutive_relation;

  void operator()(gsl::not_null<tnsr::I<DataType, 2>*> displacement,
                  gsl::not_null<Cache*> cache,
                  Tags::Displacement<2> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::ii<DataType, 2>*> strain,
                  gsl::not_null<Cache*> cache, Tags::Strain<2> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::II<DataType, 2>*> minus_stress,
                  gsl::not_null<Cache*> cache,
                  Tags::MinusStress<2> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> potential_energy_density,
                  gsl::not_null<Cache*> cache,
                  Tags::PotentialEnergyDensity<2> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::I<DataType, 2>*> fixed_source_for_displacement,
      gsl::not_null<Cache*> cache,
      ::Tags::FixedSource<Tags::Displacement<2>> /*meta*/) const;
};
}  // namespace detail

/// \cond
template <typename Registrars>
struct BentBeam;

namespace Registrars {
using BentBeam = ::Registration::Registrar<Solutions::BentBeam>;
}  // namespace Registrars
/// \endcond

/*!
 * \brief A state of pure bending of an elastic beam in 2D
 *
 * \details This solution describes a 2D slice through an elastic beam of length
 * \f$L\f$ and height \f$H\f$, centered around (0, 0), that is subject to a
 * bending moment \f$M=\int T^{xx}y\mathrm{d}y\f$ (see e.g.
 * \cite ThorneBlandford2017, Eq. 11.41c for a bending moment in 1D). The beam
 * material is characterized by an isotropic and homogeneous constitutive
 * relation \f$Y^{ijkl}\f$ in the plane-stress approximation (see
 * `Elasticity::ConstitutiveRelations::IsotropicHomogeneous`). In this scenario,
 * no body-forces \f$f_\mathrm{ext}^j\f$ act on the material, so the
 * \ref Elasticity equations reduce to \f$\nabla_i T^{ij}=0\f$, but the bending
 * moment \f$M\f$ generates the stress
 *
 * \f{align}
 * T^{xx} &= \frac{12 M}{H^3} y \\
 * T^{xy} &= 0 = T^{yy} \text{.}
 * \f}
 *
 * By fixing the rigid-body motions to
 *
 * \f[
 * \xi^x(0,y)=0 \quad \text{and} \quad \xi^y\left(\pm \frac{L}{2},0\right)=0
 * \f]
 *
 * we find that this stress is produced by the displacement field
 *
 * \f{align}
 * \xi^x&=-\frac{12 M}{EH^3}xy \\
 * \xi^y&=\frac{6 M}{EH^3}\left(x^2+\nu y^2-\frac{L^2}{4}\right)
 * \f}
 *
 * in terms of the Young's modulus \f$E\f$ and the Poisson ration \f$\nu\f$ of
 * the material. The corresponding strain \f$S_{ij}=\partial_{(i}\xi_{j)}\f$ is
 *
 * \f{align}
 * S_{xx} &= -\frac{12 M}{EH^3} y \\
 * S_{yy} &= \frac{12 M}{EH^3} \nu y \\
 * S_{xy} &= S_{yx} = 0
 * \f}
 *
 * and the potential energy stored in the entire infinitesimal slice is
 *
 * \f[
 * \int_{-L/2}^{L/2} \int_{-H/2}^{H/2} U dy\,dx = \frac{6M^2}{EH^3}L \text{.}
 * \f]
 */
template <typename Registrars = tmpl::list<Solutions::Registrars::BentBeam>>
class BentBeam : public AnalyticSolution<2, Registrars> {
 public:
  using constitutive_relation_type =
      Elasticity::ConstitutiveRelations::IsotropicHomogeneous<2>;

  struct Length {
    using type = double;
    static constexpr Options::String help{"The beam length"};
    static type lower_bound() { return 0.0; }
  };
  struct Height {
    using type = double;
    static constexpr Options::String help{"The beam height"};
    static type lower_bound() { return 0.0; }
  };
  struct BendingMoment {
    using type = double;
    static constexpr Options::String help{
        "The bending moment applied to the beam"};
    static type lower_bound() { return 0.0; }
  };
  struct Material {
    using type = constitutive_relation_type;
    static constexpr Options::String help{
        "The material properties of the beam"};
  };

  using options = tmpl::list<Length, Height, BendingMoment, Material>;
  static constexpr Options::String help{
      "A 2D slice through an elastic beam which is subject to a bending "
      "moment. The bending moment is applied along the length of the beam, "
      "i.e. the x-axis, so that the beam's left and right ends are bent "
      "towards the positive y-axis. It is measured in units of force."};

  BentBeam() = default;
  BentBeam(const BentBeam&) = default;
  BentBeam& operator=(const BentBeam&) = default;
  BentBeam(BentBeam&&) = default;
  BentBeam& operator=(BentBeam&&) = default;
  ~BentBeam() override = default;

  /// \cond
  explicit BentBeam(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(BentBeam);  // NOLINT
  /// \endcond

  BentBeam(double length, double height, double bending_moment,
           constitutive_relation_type constitutive_relation)
      : length_(length),
        height_(height),
        bending_moment_(bending_moment),
        constitutive_relation_(std::move(constitutive_relation)) {}

  double length() const { return length_; }
  double height() const { return height_; }
  double bending_moment() const { return bending_moment_; }

  const constitutive_relation_type& constitutive_relation() const override {
    return constitutive_relation_;
  }

  /// Return potential energy integrated over the whole beam material
  double potential_energy() const {
    return 6. * length_ * square(bending_moment_) /
           (cube(height_) * constitutive_relation_.youngs_modulus());
  }

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 2>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using VarsComputer = detail::BentBeamVariables<DataType>;
    typename VarsComputer::Cache cache{
        get_size(*x.begin()), VarsComputer{x, length_, height_, bending_moment_,
                                           constitutive_relation_}};
    return {cache.get_var(RequestedTags{})...};
  }

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) override {  // NOLINT
    p | length_;
    p | height_;
    p | bending_moment_;
    p | constitutive_relation_;
  }

 private:
  double length_{std::numeric_limits<double>::signaling_NaN()};
  double height_{std::numeric_limits<double>::signaling_NaN()};
  double bending_moment_{std::numeric_limits<double>::signaling_NaN()};
  constitutive_relation_type constitutive_relation_{};
};

/// \cond
template <typename Registrars>
PUP::able::PUP_ID BentBeam<Registrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond

template <typename Registrars>
bool operator==(const BentBeam<Registrars>& lhs,
                const BentBeam<Registrars>& rhs) {
  return lhs.length() == rhs.length() and lhs.height() == rhs.height() and
         lhs.bending_moment() == rhs.bending_moment() and
         lhs.constitutive_relation() == rhs.constitutive_relation();
}

template <typename Registrars>
bool operator!=(const BentBeam<Registrars>& lhs,
                const BentBeam<Registrars>& rhs) {
  return not(lhs == rhs);
}

}  // namespace Elasticity::Solutions
