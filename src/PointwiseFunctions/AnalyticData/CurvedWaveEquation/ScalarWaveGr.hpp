// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <utility>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
/// \endcond

// IWYU pragma: no_include <pup.h>

namespace CurvedScalarWave {
namespace AnalyticData {
/*!
 * \brief Analytic initial data for scalar waves in curved spacetime
 *
 * \details When evolving a scalar field propagating through curved spacetime,
 * this class provides a method to initialize the scalar-field and spacetime
 * variables using analytic solution(s) of the flat-space scalar-wave equation
 * and of the Einstein equations. Note that the coordinate profile of the scalar
 * field \f$\Psi\f$ in curved spacetime being the same as \f$\Psi\f$ in flat
 * spacetime is our primary identification, allowing it to be initialized using
 * any member class of `ScalarWave::Solutions`. We initialize \f$\Phi_i\f$ in
 * curved spacetime to the coordinate spatial derivative of \f$\Psi\f$ in flat
 * spacetime. The definition of \f$\Pi\f$ comes from requiring it to be the
 * future-directed time derivative of the scalar field in curved spacetime:
 *
 * \f{align}
 * \Pi :=& -n^a \partial_a \Psi \\
 *     =&  \frac{1}{\alpha}\left(\beta^k \Phi_k - {\partial_t\Psi}\right),\\
 *     =&  \frac{1}{\alpha}\left(\beta^k \Phi_k + {\Pi}_{\mathrm{flat}}\right),
 * \f}
 *
 * where \f$n^a\f$ is the unit normal to spatial slices of the spacetime
 * foliation, and \f${\Pi}_{\mathrm{flat}}\f$ comes from the flat spacetime
 * solution.
 */
template <typename ScalarFieldData, typename BackgroundGrData>
class ScalarWaveGr : public MarkAsAnalyticData {
  static_assert(
      ScalarFieldData::volume_dim == BackgroundGrData::volume_dim,
      "Scalar field data and background spacetime data should have the same "
      "spatial dimensionality. Currently provided template arguments do not.");

 public:
  static constexpr size_t volume_dim = ScalarFieldData::volume_dim;

  struct ScalarField {
    using type = ScalarFieldData;
    static constexpr Options::String help = {"Flat space scalar field."};
  };
  struct Background {
    using type = BackgroundGrData;
    static constexpr Options::String help = {"Background spacetime."};
  };

  using options = tmpl::list<Background, ScalarField>;
  static constexpr Options::String help{
      "A scalar field in curved background spacetime\n\n"};
  static std::string name() noexcept { return "ScalarWaveGr"; };

  // Construct from options
  ScalarWaveGr(BackgroundGrData background,
               ScalarFieldData scalar_field) noexcept
      : flat_space_scalar_wave_data_(std::move(scalar_field)),
        background_gr_data_(std::move(background)) {}

  explicit ScalarWaveGr(CkMigrateMessage* /*unused*/) noexcept {}

  ScalarWaveGr() = default;
  ScalarWaveGr(const ScalarWaveGr& /*rhs*/) = delete;
  ScalarWaveGr& operator=(const ScalarWaveGr& /*rhs*/) = delete;
  ScalarWaveGr(ScalarWaveGr&& /*rhs*/) noexcept = default;
  ScalarWaveGr& operator=(ScalarWaveGr&& /*rhs*/) noexcept = default;
  ~ScalarWaveGr() = default;

  // Tags
  template <typename DataType>
  using spacetime_tags = typename BackgroundGrData::template tags<DataType>;
  using tags = tmpl::append<spacetime_tags<DataVector>,
                            tmpl::list<Pi, Phi<volume_dim>, Psi>>;

  /// Retrieve spacetime variables
  template <
      typename DataType, typename Tag,
      Requires<tmpl::list_contains_v<spacetime_tags<DataType>, Tag>> = nullptr>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, volume_dim>& x,
                                     tmpl::list<Tag> /*meta*/) const noexcept {
    constexpr double default_initial_time = 0.;
    return {std::move(get<Tag>(background_gr_data_.variables(
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
        get<ScalarWave::Phi<volume_dim>>(flat_space_scalar_wave_data_.variables(
            x, default_initial_time,
            tmpl::list<ScalarWave::Pi, ScalarWave::Phi<volume_dim>,
                       ScalarWave::Psi>{})))};
  }
  tuples::TaggedTuple<Psi> variables(const tnsr::I<DataVector, volume_dim>& x,
                                     tmpl::list<Psi> /*meta*/) const noexcept {
    constexpr double default_initial_time = 0.;
    return {
        std::move(get<ScalarWave::Psi>(flat_space_scalar_wave_data_.variables(
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

 private:
  template <typename LocalScalarFieldData,                  // NOLINT
            typename LocalBackgroundData>                   // NOLINT
  friend bool                                               // NOLINT
  operator==(const ScalarWaveGr<LocalScalarFieldData,       // NOLINT
                                LocalBackgroundData>& lhs,  // NOLINT
             const ScalarWaveGr<LocalScalarFieldData,       // NOLINT
                                LocalBackgroundData>& rhs)  // NOLINT
      noexcept;                                             // NOLINT

  ScalarFieldData flat_space_scalar_wave_data_;
  BackgroundGrData background_gr_data_;
};

template <typename ScalarFieldData, typename BackgroundData>
bool operator!=(
    const ScalarWaveGr<ScalarFieldData, BackgroundData>& lhs,
    const ScalarWaveGr<ScalarFieldData, BackgroundData>& rhs) noexcept;

}  // namespace AnalyticData
}  // namespace CurvedScalarWave
