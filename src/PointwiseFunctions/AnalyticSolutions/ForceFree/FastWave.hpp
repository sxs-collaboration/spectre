// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
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
 * \brief An electromagnetic wave propagating into \f$+x\f$ direction in flat
 * spacetime.
 *
 * The initial data is given by \cite Komissarov2002
 *
 * \f{align*}{
 *  B^x & = 1.0 , \\
 *  B^y & = \left\{\begin{array}{ll}
 *  1.0          & \text{if } x < -0.1 \\
 *  -1.5x + 0.85 & \text{if } -0.1 \leq x \leq 0.1 \\
 *  0.7          & \text{if } x > 0.1 \\
 * \end{array}\right\}, \\
 *  B^z & = 0 , \\
 *  E^x & = E^y = 0 , \\
 *  E^z & = -B^y .
 * \f}
 *
 * The electric and magnetic fields are advected to \f$+x\f$ direction with the
 * speed of light (\f$\lambda=+1\f$), and the solution at \f$(\vec{x},t)\f$ is
 *
 * \f{align*}{
 *  E^i(x,y,z,t) & = E^i(x-t,y,z,0) , \\
 *  B^i(x,y,z,t) & = B^i(x-t,y,z,0) .
 * \f}
 *
 * Electric charge density \f$q\f$ is zero everywhere.
 *
 */
class FastWave : public evolution::initial_data::InitialData,
                 public MarkAsAnalyticSolution {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "A fast mode wave propagating +x direction in flat spacetime"};

  FastWave() = default;
  FastWave(const FastWave&) = default;
  FastWave& operator=(const FastWave&) = default;
  FastWave(FastWave&&) = default;
  FastWave& operator=(FastWave&&) = default;
  ~FastWave() override = default;

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit FastWave(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(FastWave);
  /// \endcond

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  /// @{
  /// Retrieve the EM variables at (x,t).
  static auto variables(const tnsr::I<DataVector, 3>& x, double t,
                        tmpl::list<Tags::TildeE> /*meta*/)
      -> tuples::TaggedTuple<Tags::TildeE>;

  static auto variables(const tnsr::I<DataVector, 3>& x, double t,
                        tmpl::list<Tags::TildeB> /*meta*/)
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
  // Computes the initial profile
  static DataVector initial_profile(const DataVector& coords);
  gr::Solutions::Minkowski<3> background_spacetime_{};

  friend bool operator==(const FastWave& lhs, const FastWave& rhs);
  friend bool operator!=(const FastWave& lhs, const FastWave& rhs);
};

}  // namespace ForceFree::Solutions
