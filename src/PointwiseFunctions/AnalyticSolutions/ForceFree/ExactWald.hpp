// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/KerrSchildCoords.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ForceFree::Solutions {

/*!
 * \brief An exact electrovacuum force-free solution of Maxwell's equations in
 * the Schwarzschild spacetime by Wald \cite Wald1974.
 *
 * The solution is given in terms of the electromagnetic 4-potential
 *
 * \begin{equation}
 *  A_\mu = \frac{B_0}{2}(\phi_\mu + 2a t_\mu)
 * \end{equation}
 *
 * where $B_0$ is the vector potential amplitude, $\phi^\mu = \partial_\phi$,
 * $t^\mu = \partial_t$, and $a$ is the (dimensionless) spin of the black hole.
 * The case $a=0$ is force-free outside the horizon.
 *
 * \note This solution is not force-free inside the horizon; the condition
 * $E_iB^i = 0$ is still satisfied, but $B^2 > E^2$ is not.
 *
 * In the spherical Kerr-Schild coordinates, the only nonzero component of
 * vector potential is
 *
 * \begin{equation}
 *  A_\phi = \frac{B_0}{2}r^2 \sin^2 \theta.
 * \end{equation}
 *
 * Computing magnetic fields,
 * \begin{align}
 *   \tilde{B}^r  & = \partial_\theta A_\phi = B_0 r^2 \sin\theta\cos\theta \\
 *   \tilde{B}^\theta & = - \partial_r A_\phi = -B_0 r \sin^2 \theta \\
 *   \tilde{B}^\phi   &= 0 ,
 * \end{align}
 *
 * Transformation to the Cartesian coordinates gives
 * \begin{equation}
 *  \tilde{B}^x = 0 , \quad
 *  \tilde{B}^y = 0 , \quad
 *  \tilde{B}^z = B_0 .
 * \end{equation}
 *
 * Electric fields are given by
 *
 * \begin{equation}
 *  E_i = F_{ia}n^a = \frac{1}{\alpha}(F_{i0} - F_{ij}\beta^j) .
 * \end{equation}
 *
 * We omit the derivation and write out results below:
 *  \begin{equation}
 *    \tilde{E}^x = - \frac{2 M B_0 y}{r^2}, \quad
 *    \tilde{E}^y =   \frac{2 M B_0 x}{r^2}, \quad
 *    \tilde{E}^z = 0
 *  \end{equation}
 *
 * Note that $\tilde{B}^i \equiv \sqrt{\gamma}B^i$, $\tilde{E}^i \equiv
 * \sqrt{\gamma}E^i$, and $\gamma = 1 + 2M/r$ in the (Cartesian) Kerr-Schild
 * coordinates.  We use $M=1$ Schwarzschild black hole in the Kerr-Schild
 * coordinates (see the documentation of gr::Solutions::KerrSchild).
 *
 */
class ExactWald : public evolution::initial_data::InitialData,
                  public MarkAsAnalyticSolution {
 public:
  struct MagneticFieldAmplitude {
    using type = double;
    static constexpr Options::String help = {
        "Amplitude of magnetic field along z axis."};
  };

  using options = tmpl::list<MagneticFieldAmplitude>;

  static constexpr Options::String help{
      "Exact vacuum solution of Maxwell's equations in Schwarzschild BH "
      "spacetime by Wald (1974)."};

  ExactWald() = default;
  ExactWald(const ExactWald&) = default;
  ExactWald& operator=(const ExactWald&) = default;
  ExactWald(ExactWald&&) = default;
  ExactWald& operator=(ExactWald&&) = default;
  ~ExactWald() override = default;

  explicit ExactWald(double magnetic_field_amplitude);

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit ExactWald(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ExactWald);
  /// \endcond

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  /// @{
  /// Retrieve the EM variables at (x,t).
  auto variables(const tnsr::I<DataVector, 3>& x, double t,
                 tmpl::list<Tags::TildeE> /*meta*/) const
      -> tuples::TaggedTuple<Tags::TildeE>;

  auto variables(const tnsr::I<DataVector, 3>& x, double t,
                 tmpl::list<Tags::TildeB> /*meta*/) const
      -> tuples::TaggedTuple<Tags::TildeB>;

  static auto variables(const tnsr::I<DataVector, 3>& x, double t,
                        tmpl::list<Tags::TildePsi> /*meta*/)
      -> tuples::TaggedTuple<Tags::TildePsi>;

  static auto variables(const tnsr::I<DataVector, 3>& x, double t,
                        tmpl::list<Tags::TildePhi> /*meta*/)
      -> tuples::TaggedTuple<Tags::TildePhi>;

  static auto variables(const tnsr::I<DataVector, 3>& x, double t,
                        tmpl::list<Tags::TildeQ> /*meta*/)
      -> tuples::TaggedTuple<Tags::TildeQ>;
  /// @}

  /// Retrieve a collection of EM variables at `(x, t)`
  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataVector, 3>& x,
                                         const double t,
                                         tmpl::list<Tags...> /*meta*/) const {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {get<Tags>(variables(x, t, tmpl::list<Tags>{}))...};
  }

  /// Retrieve the metric variables
  template <typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataVector, 3>& x, double t,
                                     tmpl::list<Tag> /*meta*/) const {
    return background_spacetime_.variables(x, t, tmpl::list<Tag>{});
  }

 private:
  gr::Solutions::KerrSchild background_spacetime_{
      1.0, {{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}};
  double magnetic_field_amplitude_ =
      std::numeric_limits<double>::signaling_NaN();

  friend bool operator==(const ExactWald& lhs, const ExactWald& rhs);
};

bool operator!=(const ExactWald& lhs, const ExactWald& rhs);

}  // namespace ForceFree::Solutions
