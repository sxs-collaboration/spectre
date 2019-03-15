// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"     // IWYU pragma: keep
#include "Elliptic/Systems/Xcts/Tags.hpp"       // IWYU pragma: keep
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Xcts {
namespace Solutions {

/*!
 * \brief A constant density star in general relativity
 *
 * \details This solution describes a star with constant density \f$\rho_0\f$
 * that extends to a (conformal) radius \f$R\f$. It solves the XCTS Hamiltonian
 * constraint that reduces to the non-linear elliptic equation
 * \f[
 * \Delta^2\psi+2\pi\rho\psi^5=0
 * \f]
 * for the conformal factor \f$\psi\f$ (see `Xcts`) under the following
 * assumptions \cite Baumgarte2006ug :
 *
 * - Time-symmetry \f$K_{ij}=0\f$
 * - Conformal flatness \f$\overline{\gamma}=\delta\f$, so \f$\Delta\f$ is the
 * flat-space Laplacian
 * - Spherical symmetry
 *
 * Imposing boundary conditions
 * \f[
 * \frac{\partial\psi}{\partial r}=0 \quad \text{for} \quad r=0\\
 * \psi\rightarrow 1 \quad \text{for} \quad r\rightarrow\infty
 * \f]
 * and considering the energy density
 * \f[
 * \rho(r\leq R)=\rho_0 \quad \text{and} \quad \rho(r>R)=0
 * \f]
 * of the star the authors of \cite Baumgarte2006ug find the solution
 * \f[
 * \psi(r\leq R)=C u_\alpha(r) \quad \text{and}
 * \quad \psi(r>R)=\frac{\beta}{r} + 1
 * \f]
 * with \f$C=(2\pi\rho_0/3)^{-1/4}\f$, the Sobolev functions
 * \f[
 * u_\alpha(r)=\sqrt{\frac{\alpha R}{r^2+(\alpha R)^2}}
 * \f]
 * and real parameters \f$\alpha\f$ and \f$\beta\f$ that are determined by
 * the following relations:
 * \f[
 * \rho_0 R^2=\frac{3}{2\pi}f^2(\alpha) \quad \text{with}
 * \quad f(\alpha)=\frac{\alpha^5}{(1+\alpha^2)^3} \\
 * \frac{\beta}{R} + 1 = C u_\alpha(R)
 * \f]
 *
 * This solution is described in detail in \cite Baumgarte2006ug since it
 * exhibits the non-uniqueness properties that are typical for the XCTS system.
 * In the simple case of the constant-density star the non-uniqueness is
 * apparent from the function \f$f(\alpha)\f$, which has two solutions for any
 * \f$\rho_0\f$ smaller than a critical density
 * \f[
 * \rho_\mathrm{crit}=\frac{3}{2\pi R^2}\frac{5^2}{6^6}
 * \approx\frac{0.0320}{R^2} \text{,}
 * \f]
 * a unique solution for \f$\rho_0=\rho_\mathrm{crit}\f$ and no solutions
 * above the critical density \cite Baumgarte2006ug . The authors identify the
 * \f$\alpha < \alpha_\mathrm{crit}=\sqrt{5}\f$ and \f$\alpha >
 * \alpha_\mathrm{crit}\f$ branches of solutions with the strong-field and
 * weak-field regimes, respectively (see \cite Baumgarte2006ug for details).
 * In this implementation we compute the weak-field solution by choosing the
 * \f$\alpha > \alpha_\mathrm{crit}\f$ that corresponds to the density
 * \f$\rho_0\f$ of the star. Therefore, we supply initial data
 * \f$\psi_\mathrm{init}=1\f$ so that a nonlinear iterative numerical solver
 * will converge to the same weak-field solution.
 */
class ConstantDensityStar {
 private:
  struct Density {
    using type = double;
    static constexpr OptionString help{"The constant density within the star"};
    static double lower_bound() noexcept { return 0.; }
  };
  struct Radius {
    using type = double;
    static constexpr OptionString help{"The conformal radius of the star"};
    static double lower_bound() noexcept { return 0.; }
  };

 public:
  using options = tmpl::list<Density, Radius>;
  static constexpr OptionString help{
      "A constant density star in general relativity"};

  ConstantDensityStar() = default;
  ConstantDensityStar(const ConstantDensityStar&) noexcept = delete;
  ConstantDensityStar& operator=(const ConstantDensityStar&) noexcept = delete;
  ConstantDensityStar(ConstantDensityStar&&) noexcept = default;
  ConstantDensityStar& operator=(ConstantDensityStar&&) noexcept = default;
  ~ConstantDensityStar() noexcept = default;

  ConstantDensityStar(double density, double radius,
                      const OptionContext& context = {});

  double density() const noexcept { return density_; }
  double radius() const noexcept { return radius_; }

  // @{
  /// Retrieve variable at coordinates `x`
  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<Xcts::Tags::ConformalFactor<DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::Initial<
                     Xcts::Tags::ConformalFactor<DataType>>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          ::Tags::Initial<Xcts::Tags::ConformalFactor<DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::Initial<Xcts::Tags::ConformalFactorGradient<
                     3, Frame::Inertial, DataType>>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<::Tags::Initial<
          Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial, DataType>>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<
          ::Tags::Source<Xcts::Tags::ConformalFactor<DataType>>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          ::Tags::Source<Xcts::Tags::ConformalFactor<DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::Source<Xcts::Tags::ConformalFactorGradient<
                     3, Frame::Inertial, DataType>>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<::Tags::Source<
          Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial, DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<gr::Tags::EnergyDensity<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<gr::Tags::EnergyDensity<DataType>>;
  // @}

  /// Retrieve a collection of variables at coordinates `x`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  double density_ = std::numeric_limits<double>::signaling_NaN();
  double radius_ = std::numeric_limits<double>::signaling_NaN();
  double alpha_ = std::numeric_limits<double>::signaling_NaN();
};

bool operator==(const ConstantDensityStar& /*lhs*/,
                const ConstantDensityStar& /*rhs*/) noexcept;

bool operator!=(const ConstantDensityStar& lhs,
                const ConstantDensityStar& rhs) noexcept;

}  // namespace Solutions
}  // namespace Xcts
