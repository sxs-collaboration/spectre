// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
// IWYU pragma: no_forward_declare Tensor
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ForceFree::Solutions {

/*!
 * \brief Alfven wave propagating along \f$x\f$ direction in flat spacetime with
 * the wave speed \f$\mu\f$.
 *
 * This test problem was introduced in \cite Komissarov2004 with $\mu=0$.
 *
 * In the wave frame (with prime superscript), the stationary solution is given
 * by
 *
 * \f{align*}
 *  B'_x & = B'_y = 1.0 ,\\
 *  B'_z(x') & = \left\{\begin{array}{ll}
 *      1.0         & \text{if } x' < -0.1   \\
 *      1.15 + 0.15 \sin (5\pi x') & \text{if } -0.1 < x' < 0.1 \\
 *      1.3         & \text{if } x' > 0.1 \end{array}\right\} ,\\
 *      E'_x & = -B'_z ,\\
 *      E'_y & = 0 \\
 *      E'_z & = 1.0
 * \f}
 *
 * and
 *
 * \f{align*}
 *  q'   & = \left\{\begin{array}{ll}
 *      - 0.75 \pi \cos (5 \pi x')  & \text{if } -0.1 < x' < 0.1 \\
 *      0         & \text{otherwise}    \end{array}\right\} , \\
 *  J_x' & = 0 , \\
 *  J_y' & = \left\{\begin{array}{ll}
 *      - 0.75 \pi \cos (5 \pi x')  & \text{if } -0.1 < x' < 0.1 \\
 *      0         & \text{otherwise} \end{array}\right\} , \\
 *  J_z' & = 0 .
 * \f}
 *
 * Applying the Lorentz transformation, electromagnetic fields and 4-current in
 * the grid frame at \f$t=0\f$ are given by
 *
 * \f{align*}
 *  E_x(x) & = E'_x(\gamma x) , \\
 *  E_y(x) & = \gamma[E'_y(\gamma x) + \mu B'_z(\gamma x)] , \\
 *  E_z(x) & = \gamma[E'_z(\gamma x) - \mu B'_y(\gamma x)] , \\
 *  B_x(x) & = B'_x(\gamma x), \\
 *  B_y(x) & = \gamma[ B'_y(\gamma x) - \mu E'_z(\gamma x) ] , \\
 *  B_z(x) & = \gamma[ B'_z(\gamma x) + \mu E'_y(\gamma x) ] .
 * \f}
 *
 * and
 *
 * \f{align*}
 *  q(x)    & = \gamma q'(\gamma x) , \\
 *  J_x(x)  & = \gamma \mu q'(\gamma x) , \\
 *  J_y(x)  & = J_y'(\gamma x) , \\
 *  J_z(x)  & = 0 .
 * \f}
 *
 * The wave speed can be chosen any value $-1 < \mu < 1$, and the solution at
 * time $t$ is $f(x,t) = f(x-\mu t, 0)$ for any physical quantities.
 *
 */
class AlfvenWave : public evolution::initial_data::InitialData,
                   public MarkAsAnalyticSolution {
 public:
  /// The wave speed
  struct WaveSpeed {
    using type = double;
    static constexpr Options::String help = {
        "The wave speed along x direction"};
    static type lower_bound() { return -1.0; }
    static type upper_bound() { return 1.0; }
  };

  using options = tmpl::list<WaveSpeed>;

  static constexpr Options::String help{
      "Alfven wave propagating along x direction in flat spacetime with the "
      "wave speed mu"};

  AlfvenWave() = default;
  AlfvenWave(const AlfvenWave&) = default;
  AlfvenWave& operator=(const AlfvenWave&) = default;
  AlfvenWave(AlfvenWave&&) = default;
  AlfvenWave& operator=(AlfvenWave&&) = default;
  ~AlfvenWave() override = default;

  AlfvenWave(double wave_speed, const Options::Context& context = {});

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit AlfvenWave(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(AlfvenWave);
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

  auto variables(const tnsr::I<DataVector, 3>& x, double t,
                 tmpl::list<Tags::TildeQ> /*meta*/) const
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
  static DataVector wave_profile(const DataVector& x_prime);
  static DataVector charge_density(const DataVector& x_prime);
  double wave_speed_ = std::numeric_limits<double>::signaling_NaN();
  double lorentz_factor_ = std::numeric_limits<double>::signaling_NaN();
  gr::Solutions::Minkowski<3> background_spacetime_{};

  friend bool operator==(const AlfvenWave& lhs, const AlfvenWave& rhs);
};

bool operator!=(const AlfvenWave& lhs, const AlfvenWave& rhs);

}  // namespace ForceFree::Solutions
