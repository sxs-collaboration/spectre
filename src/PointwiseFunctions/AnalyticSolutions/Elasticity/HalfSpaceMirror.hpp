// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/AnalyticSolution.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Elasticity::Solutions {

namespace detail {
template <typename DataType>
struct HalfSpaceMirrorVariables {
  using Cache = CachedTempBuffer<HalfSpaceMirrorVariables,
                                 Tags::Displacement<3>, Tags::Strain<3>,
                                 ::Tags::FixedSource<Tags::Displacement<3>>>;

  const tnsr::I<DataType, 3>& x;
  const double beam_width;
  const ConstitutiveRelations::IsotropicHomogeneous<3>& constitutive_relation;
  const size_t integration_intervals;
  const double absolute_tolerance;
  const double relative_tolerance;

  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> displacement,
                  gsl::not_null<Cache*> cache,
                  Tags::Displacement<3> /*meta*/) const noexcept;
  void operator()(gsl::not_null<tnsr::ii<DataType, 3>*> strain,
                  gsl::not_null<Cache*> cache, Tags::Strain<3> /*meta*/) const
      noexcept;
  void operator()(
      gsl::not_null<tnsr::I<DataType, 3>*> fixed_source_for_displacement,
      gsl::not_null<Cache*> cache,
      ::Tags::FixedSource<Tags::Displacement<3>> /*meta*/) const noexcept;
};
}  // namespace detail

/// \cond
template <typename Registrars>
struct HalfSpaceMirror;

namespace Registrars {
using HalfSpaceMirror = ::Registration::Registrar<Solutions::HalfSpaceMirror>;
}  // namespace Registrars
/// \endcond

/*!
 * \brief The solution for a half-space mirror deformed by a laser beam.
 *
 * \details This solution is mapping (via the fluctuation dissipation theorem)
 * thermal noise to an elasticity problem where a normally incident and
 * axisymmetric laser beam with a Gaussian beam profile acts on the face of a
 * semi-infinite mirror. Here we assume the face to be at \f$z = 0\f$ and the
 * material to extend to \f$+\infty\f$ in the z-direction as well as for the
 * mirror diameter to be comparatively large to the `beam width`. The mirror
 * material is characterized by an isotropic homogeneous constitutive relation
 * \f$Y^{ijkl}\f$ (see
 * `Elasticity::ConstitutiveRelations::IsotropicHomogeneous`). In this scenario,
 * the auxiliary elastic problem has an applied pressure distribution equal to
 * the laser beam intensity profile \f$p(r)\f$ (see Eq. (11.94) and Eq. (11.95)
 * in \cite ThorneBlandford2017 with F = 1 and the time dependency dropped)
 *
 * \f{align}
 * T^{zr} &= T^{rz} = 0 \\
 * T^{zz} &= p(r) = \frac{e^{-\frac{r^2}{r_0^2}}}{\pi r_0^2}\text{.}
 * \f}
 *
 * in the form of a Neumann boundary condition to the face of the mirror. We
 * find that this stress in cylinder coordinates is produced by the displacement
 * field
 *
 * \f{align}
 * \xi_{r} &= \frac{1}{2 \mu} \int_0^{\infty} dk J_1(kr)e^{(-kz)}\left(1 -
 * \frac{\lambda + 2\mu}{\lambda + \mu} + kz \right) \tilde{p}(k) \\
 * \xi_{\phi} &= 0 \\
 * \xi_{z} &=  \frac{1}{2 \mu} \int_0^{\infty} dk J_0(kr)e^{(-kz)}\left(1 +
 * \frac{\mu}{\lambda + \mu} + kz \right) \tilde{p}(k)
 * \f}
 *
 * and the strain
 *
 * \f{align}
 * \Theta &= \frac{1}{2 \mu} \int_0^{\infty} dk
 * J_0(kr) k e^{(-kz)}\left(\frac{-2\mu}{\lambda + \mu}\right) \tilde{p}(k) \\
 * S_{rr} &= \Theta - S_{\phi\phi} - S_{zz} \\
 * S_{\phi\phi} &= \frac{\xi_{r}}{r} \\
 * S_{(rz)} &= -\frac{1}{2 \mu} \int_0^{\infty} dk J_1(kr) k e^{(-kz)}\left(kz
 * \right) \tilde{p}(k) \\
 * S_{zz} &= \frac{1}{2 \mu} \int_0^{\infty} dk
 * J_0(kr) k e^{(-kz)}\left(-\frac{\mu}{\lambda + \mu} - kz \right) \tilde{p}(k)
 * \f}
 *
 * (see Eqs. (11 a) - (11 c) and (13 a) - (13 e), with (13 c) swapped in favor
 * of (12 c) in \cite Lovelace2007tn), where \f$\tilde{p}(k)= \frac{1}{2\pi}
 * e^{-(\frac{kr_0}{2})^2}\f$ is the Hankel-Transform of the lasers intensity
 * profile and \f$ \Theta = \mathrm{Tr}(S)\f$ the materials expansion.
 *
 */
template <typename Registrars =
              tmpl::list<Solutions::Registrars::HalfSpaceMirror>>
class HalfSpaceMirror : public AnalyticSolution<3, Registrars> {
 public:
  using constitutive_relation_type =
      Elasticity::ConstitutiveRelations::IsotropicHomogeneous<3>;

  struct BeamWidth {
    using type = double;
    static constexpr Options::String help{
        "The lasers beam width r_0 with FWHM = 2*sqrt(ln 2)*r_0"};
    static type lower_bound() noexcept { return 0.0; }
  };

  struct Material {
    using type = constitutive_relation_type;
    static constexpr Options::String help{
        "The material properties of the beam"};
  };

  struct IntegrationIntervals {
    using type = size_t;
    static constexpr Options::String help{
        "Workspace size for numerical integrals. Increase if integrals fail to "
        "reach the prescribed tolerance at large distances relative to the "
        "beam width. The suggested values for workspace size and tolerances "
        "should accommodate distances of up to ~100 beam widths."};
    static type lower_bound() noexcept { return 1; }
    static type suggested_value() noexcept { return 350; }
  };

  struct AbsoluteTolerance {
    using type = double;
    static constexpr Options::String help{
        "Absolute tolerance for numerical integrals"};
    static type lower_bound() noexcept { return 0.; }
    static type suggested_value() noexcept { return 1e-12; }
  };

  struct RelativeTolerance {
    using type = double;
    static constexpr Options::String help{
        "Relative tolerance for numerical integrals"};
    static type lower_bound() noexcept { return 0.; }
    static type upper_bound() noexcept { return 1.; }
    static type suggested_value() noexcept { return 1e-10; }
  };

  using options = tmpl::list<BeamWidth, Material, IntegrationIntervals,
                             AbsoluteTolerance, RelativeTolerance>;
  static constexpr Options::String help{
      "A semi-infinite mirror on which a laser introduces stress perpendicular "
      "to the mirrors surface."};

  HalfSpaceMirror() = default;
  HalfSpaceMirror(const HalfSpaceMirror&) noexcept = default;
  HalfSpaceMirror& operator=(const HalfSpaceMirror&) noexcept = default;
  HalfSpaceMirror(HalfSpaceMirror&&) noexcept = default;
  HalfSpaceMirror& operator=(HalfSpaceMirror&&) noexcept = default;
  ~HalfSpaceMirror() noexcept override = default;

  /// \cond
  explicit HalfSpaceMirror(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(HalfSpaceMirror);  // NOLINT
  /// \endcond

  HalfSpaceMirror(double beam_width,
                  constitutive_relation_type constitutive_relation,
                  size_t integration_intervals = 350,
                  double absolute_tolerance = 1e-12,
                  double relative_tolerance = 1e-10) noexcept
      : beam_width_(beam_width),
        constitutive_relation_(std::move(constitutive_relation)),
        integration_intervals_(integration_intervals),
        absolute_tolerance_(absolute_tolerance),
        relative_tolerance_(relative_tolerance) {}

  double beam_width() const noexcept { return beam_width_; }
  size_t integration_intervals() const noexcept {
    return integration_intervals_;
  }
  double absolute_tolerance() const noexcept { return absolute_tolerance_; }
  double relative_tolerance() const noexcept { return relative_tolerance_; }

  const constitutive_relation_type& constitutive_relation() const
      noexcept override {
    return constitutive_relation_;
  }

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<RequestedTags...> /*meta*/) const noexcept {
    for (size_t i = 0; i < get_size(get<2>(x)); i++) {
      if (UNLIKELY(get_element(get<2>(x), i) < 0)) {
        ERROR(
            "The HalfSpaceMirror solution is not defined for negative values "
            "of z.");
      }
    }
    using VarsComputer = detail::HalfSpaceMirrorVariables<DataType>;
    typename VarsComputer::Cache cache{
        get_size(*x.begin()),
        VarsComputer{x, beam_width_, constitutive_relation_,
                     integration_intervals_, absolute_tolerance_,
                     relative_tolerance_}};
    return {cache.get_var(RequestedTags{})...};
  }

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) noexcept override {  // NOLINT
    p | beam_width_;
    p | constitutive_relation_;
    p | integration_intervals_;
    p | absolute_tolerance_;
    p | relative_tolerance_;
  }

 private:
  double beam_width_{std::numeric_limits<double>::signaling_NaN()};
  constitutive_relation_type constitutive_relation_{};
  size_t integration_intervals_{std::numeric_limits<size_t>::max()};
  double absolute_tolerance_{std::numeric_limits<double>::signaling_NaN()};
  double relative_tolerance_{std::numeric_limits<double>::signaling_NaN()};
};

/// \cond
template <typename Registrars>
PUP::able::PUP_ID HalfSpaceMirror<Registrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond

template <typename Registrars>
bool operator==(const HalfSpaceMirror<Registrars>& lhs,
                const HalfSpaceMirror<Registrars>& rhs) noexcept {
  return lhs.beam_width() == rhs.beam_width() and
         lhs.constitutive_relation() == rhs.constitutive_relation() and
         lhs.integration_intervals() == rhs.integration_intervals() and
         lhs.absolute_tolerance() == rhs.absolute_tolerance() and
         lhs.relative_tolerance() == rhs.relative_tolerance();
}

template <typename Registrars>
bool operator!=(const HalfSpaceMirror<Registrars>& lhs,
                const HalfSpaceMirror<Registrars>& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Elasticity::Solutions
