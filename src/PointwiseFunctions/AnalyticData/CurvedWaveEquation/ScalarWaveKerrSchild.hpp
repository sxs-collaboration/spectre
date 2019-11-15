// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/RegularSphericalWave.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
namespace Tags {
template <typename Tag>
struct dt;
}  // namespace Tags
/// \endcond

// IWYU pragma: no_include <pup.h>

namespace CurvedScalarWave {
namespace AnalyticData {
/*!
 * \brief Analytic initial data for scalar waves in \f$3+1\f$D Kerr spacetime
 *
 * \details When evolving a scalar field propagating through curved spacetime,
 * this class provides methods to initialize the scalar-field and spacetime
 * variables using analytic solutions to the flat-space scalar-wave equation
 * and a Kerr background spacetime. That the coordinate profile of the scalar
 * field \f$\psi\f$ in curved spacetime is the same as that in flat spacetime
 * is our primary identification, allowing it to be initialized using any
 * member class of `ScalarWave::Solutions`. Solving constraints of the
 * `CurvedScalarWave` system, we also identify the coordinate spatial derivative
 * of the scalar field \f$\Phi_i\f$ in curved spacetime with the same in flat
 * spacetime. The definition of \f$\Pi\f$ comes from requiring it to be the
 * future-directed time derivative of the scalar field:
 *
 * \f{align}
 * \Pi := -n^a \partial_a \psi
 *     =  \frac{1}{N}\left(N^k \Phi_k - {\partial_t\psi}\right),
 * \f}
 *
 * where \f$n^a\f$ is the unit normal to spatial slices of the spacetime
 * foliation.
 */
template <typename ScalarFieldSolution>
class ScalarWaveKerrSchild : public MarkAsAnalyticData {
 public:
  static constexpr size_t volume_dim = 3;

  struct WaveVector {
    using type = std::array<double, volume_dim>;
    static constexpr OptionString help = {
        "The propagation vector for the plane wave."};
  };
  struct WaveCenter {
    using type = std::array<double, volume_dim>;
    static constexpr OptionString help = {
        "The initial center of the plane wave."};
  };
  struct WaveProfile {
    using type = std::unique_ptr<MathFunction<1>>;
    static constexpr OptionString help = {
        "The radial profile of the spherical wave."};
  };
  struct BlackHoleMass {
    using type = double;
    static constexpr OptionString help = {"Mass of the black hole"};
    static type default_value() noexcept { return 1.; }
    static type lower_bound() noexcept { return 0.; }
  };
  struct BlackHoleSpin {
    using type = std::array<double, volume_dim>;
    static constexpr OptionString help = {
        "The [x,y,z] dimensionless spin of the black hole"};
    static type default_value() noexcept { return {{0., 0., 0.}}; }
  };
  struct BlackHoleCenter {
    using type = std::array<double, volume_dim>;
    static constexpr OptionString help = {
        "The [x,y,z] center of the black hole"};
    static type default_value() noexcept { return {{0., 0., 0.}}; }
  };
  using options = tmpl::conditional_t<
      std::is_same_v<ScalarFieldSolution,
                     ScalarWave::Solutions::RegularSphericalWave>,
      tmpl::list<BlackHoleMass, BlackHoleSpin, BlackHoleCenter, WaveProfile>,
      tmpl::list<BlackHoleMass, BlackHoleSpin, BlackHoleCenter, WaveVector,
                 WaveCenter, WaveProfile>>;
  static constexpr OptionString help{
      "A scalar wave with a Kerr black hole (in Kerr-Schild coordinates)\n"
      " as background\n\n"};
  static std::string name() noexcept { return "ScalarWaveKerr"; };

  // Enable or disable this constructor based on ScalarFieldSolution
  template <typename Dummy = ScalarWave::Solutions::RegularSphericalWave,
            Requires<std::is_same<ScalarFieldSolution, Dummy>::value> = nullptr>
  explicit ScalarWaveKerrSchild(
      double mass, std::array<double, volume_dim> dimensionless_spin,
      std::array<double, volume_dim> center,
      std::unique_ptr<MathFunction<1>> profile,
      const OptionContext& context = {}) noexcept;

  // Enable or disable this constructor based on ScalarFieldSolution
  template <typename Dummy = ScalarWave::Solutions::PlaneWave<volume_dim>,
            Requires<std::is_same<ScalarFieldSolution, Dummy>::value> = nullptr>
  explicit ScalarWaveKerrSchild(
      double mass, std::array<double, volume_dim> dimensionless_spin,
      std::array<double, volume_dim> center,
      std::array<double, volume_dim> wave_vector,
      std::array<double, volume_dim> wave_center,
      std::unique_ptr<MathFunction<1>> profile,
      const OptionContext& context = {}) noexcept;

  explicit ScalarWaveKerrSchild(CkMigrateMessage* /*unused*/) noexcept {}

  ScalarWaveKerrSchild() = default;
  ScalarWaveKerrSchild(const ScalarWaveKerrSchild& /*rhs*/) = delete;
  ScalarWaveKerrSchild& operator=(const ScalarWaveKerrSchild& /*rhs*/) = delete;
  ScalarWaveKerrSchild(ScalarWaveKerrSchild&& /*rhs*/) noexcept = default;
  ScalarWaveKerrSchild& operator=(ScalarWaveKerrSchild&& /*rhs*/) noexcept =
      default;
  ~ScalarWaveKerrSchild() = default;

  // Tags
  template <typename DataType>
  using spacetime_tags = gr::Solutions::KerrSchild::tags<DataType>;
  using tags = tmpl::append<spacetime_tags<DataVector>,
                            tmpl::list<Pi, Phi<volume_dim>, Psi>>;

  /// Retrieve spacetime variables
  template <
      typename DataType, typename Tag,
      Requires<tmpl::list_contains_v<spacetime_tags<DataType>, Tag>> = nullptr>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, volume_dim>& x,
                                     tmpl::list<Tag> /*meta*/) const noexcept {
    constexpr double default_initial_time = 0.;
    return {std::move(get<Tag>(kerr_schild_soln_.variables(
        x, default_initial_time, spacetime_tags<DataType>{})))};
  }

  /// Retrieve scalar wave variables
  tuples::TaggedTuple<Pi> variables(const tnsr::I<DataVector, volume_dim>& x,
                                    tmpl::list<Pi> /*meta*/) const noexcept;
  tuples::TaggedTuple<Phi<volume_dim>> variables(
      const tnsr::I<DataVector, volume_dim>& x,
      tmpl::list<Phi<volume_dim>> /*meta*/) const noexcept {
    constexpr double default_initial_time = 0.;
    return {std::move(
        get<ScalarWave::Phi<volume_dim>>(flat_space_wave_soln_.variables(
            x, default_initial_time,
            tmpl::list<ScalarWave::Pi, ScalarWave::Phi<volume_dim>,
                       ScalarWave::Psi>{})))};
  }
  tuples::TaggedTuple<Psi> variables(const tnsr::I<DataVector, volume_dim>& x,
                                     tmpl::list<Psi> /*meta*/) const noexcept {
    constexpr double default_initial_time = 0.;
    return {std::move(get<ScalarWave::Psi>(flat_space_wave_soln_.variables(
        x, default_initial_time,
        tmpl::list<ScalarWave::Pi, ScalarWave::Phi<volume_dim>,
                   ScalarWave::Psi>{})))};
  }

  // Retrieve one or more tags
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, volume_dim>& x,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    static_assert(sizeof...(Tags) > 1,
                  "This generic template will recurse infinitely if only one "
                  "tag is being retrieved through it.");

    static_assert(tmpl2::flat_all_v<tmpl::list_contains_v<tags, Tags>...>,
                  "At least one of the requested tags is not supported.");

    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& p) noexcept;  // NOLINT

  SPECTRE_ALWAYS_INLINE double mass() const noexcept {
    return kerr_schild_soln_.mass();
  }
  SPECTRE_ALWAYS_INLINE const std::array<double, volume_dim>& center() const
      noexcept {
    return kerr_schild_soln_.center();
  }
  SPECTRE_ALWAYS_INLINE const std::array<double, volume_dim>&
  dimensionless_spin() const noexcept {
    return kerr_schild_soln_.dimensionless_spin();
  }

 private:
  template <typename LocalScalarFieldSolution>
  friend bool operator==(
      const ScalarWaveKerrSchild<LocalScalarFieldSolution>& lhs,
      const ScalarWaveKerrSchild<LocalScalarFieldSolution>& rhs) noexcept;

  ScalarFieldSolution flat_space_wave_soln_;
  gr::Solutions::KerrSchild kerr_schild_soln_;
};

template <typename LocalScalarFieldSolution>
bool operator!=(
    const ScalarWaveKerrSchild<LocalScalarFieldSolution>& lhs,
    const ScalarWaveKerrSchild<LocalScalarFieldSolution>& rhs) noexcept;

}  // namespace AnalyticData
}  // namespace CurvedScalarWave
