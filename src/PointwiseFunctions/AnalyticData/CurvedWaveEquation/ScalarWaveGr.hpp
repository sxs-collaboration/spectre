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
#include "Utilities/TMPL.hpp"

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
 * this class provides a method to initialize the scalar field and spacetime
 * variables using
 *
 * 1. analytic solution(s) or data of the flat or curved scalar wave equation
 * for the evolution variables
 * 2. solutions of the Einstein equations for the spacetime background.
 *
 * If the scalar field initial data returns `CurvedScalarWave` tags, \f$\Psi\f$,
 * \f$\Pi\f$ and \f$\Phi_i\f$ will simply be forwarded from the initial data
 * class. Alternatively, the scalar field initial data can be provided using any
 * member class of `ScalarWave::Solutions` which return `ScalarWave` tags. In
 * this case, \f$\Phi_i\f$ and \f$\Psi\f$ will also be forwarded but
 * \f$\Pi\f$ will be adjusted to account for the curved background. Its
 * definition comes from requiring it to be the future-directed time derivative
 * of the scalar field in curved spacetime:
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
  static_assert(tmpl::list_contains_v<typename ScalarFieldData::tags,
                                      ScalarWave::Tags::Psi> or
                    tmpl::list_contains_v<typename ScalarFieldData::tags,
                                          CurvedScalarWave::Tags::Psi>,
                "The scalar field data needs to be able to return either the "
                "ScalarWave or the CurvedScalarWave evolved variables");

 public:
  static constexpr size_t volume_dim = ScalarFieldData::volume_dim;
  static constexpr bool is_curved =
      tmpl::list_contains_v<typename ScalarFieldData::tags,
                            CurvedScalarWave::Tags::Psi>;
  struct ScalarField {
    using type = ScalarFieldData;
    static constexpr Options::String help = {"The scalar field system data."};
  };
  struct Background {
    using type = BackgroundGrData;
    static constexpr Options::String help = {"Background spacetime."};
  };

  using options = tmpl::list<Background, ScalarField>;
  static constexpr Options::String help{
      "A scalar field in curved background spacetime\n\n"};
  static std::string name() { return "ScalarWaveGr"; };

  // Construct from options
  ScalarWaveGr(BackgroundGrData background, ScalarFieldData scalar_field)
      : scalar_wave_data_(std::move(scalar_field)),
        background_gr_data_(std::move(background)) {}

  explicit ScalarWaveGr(CkMigrateMessage* /*unused*/) {}

  ScalarWaveGr() = default;
  ScalarWaveGr(const ScalarWaveGr& /*rhs*/) = delete;
  ScalarWaveGr& operator=(const ScalarWaveGr& /*rhs*/) = delete;
  ScalarWaveGr(ScalarWaveGr&& /*rhs*/) = default;
  ScalarWaveGr& operator=(ScalarWaveGr&& /*rhs*/) = default;
  ~ScalarWaveGr() = default;

  // Tags
  template <typename DataType>
  using spacetime_tags = typename BackgroundGrData::template tags<DataType>;
  using InitialDataPsi =
      tmpl::conditional_t<is_curved, CurvedScalarWave::Tags::Psi,
                          ScalarWave::Tags::Psi>;
  using InitialDataPi =
      tmpl::conditional_t<is_curved, CurvedScalarWave::Tags::Pi,
                          ScalarWave::Tags::Pi>;
  using InitialDataPhi =
      tmpl::conditional_t<is_curved, CurvedScalarWave::Tags::Phi<volume_dim>,
                          ScalarWave::Tags::Phi<volume_dim>>;
  using evolved_field_vars_tags =
      tmpl::list<InitialDataPsi, InitialDataPi, InitialDataPhi>;
  using tags =
      tmpl::append<spacetime_tags<DataVector>,
                   tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<volume_dim>>>;

  /// Retrieve spacetime variables
  template <
      typename DataType, typename Tag,
      Requires<tmpl::list_contains_v<spacetime_tags<DataType>, Tag>> = nullptr>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, volume_dim>& x,
                                     tmpl::list<Tag> /*meta*/) const {
    constexpr double default_initial_time = 0.;
    return {std::move(get<Tag>(background_gr_data_.variables(
        x, default_initial_time, spacetime_tags<DataType>{})))};
  }

  tuples::TaggedTuple<Tags::Psi> variables(
      const tnsr::I<DataVector, volume_dim>& x,
      tmpl::list<Tags::Psi> /*meta*/) const {
    constexpr double default_initial_time = 0.;
    return {std::move(get<InitialDataPsi>(scalar_wave_data_.variables(
        x, default_initial_time, evolved_field_vars_tags{})))};
  }

  /// Retrieve scalar wave variables
  tuples::TaggedTuple<Tags::Pi> variables(
      const tnsr::I<DataVector, volume_dim>& x,
      tmpl::list<Tags::Pi> /*meta*/) const;

  tuples::TaggedTuple<Tags::Phi<volume_dim>> variables(
      const tnsr::I<DataVector, volume_dim>& x,
      tmpl::list<Tags::Phi<volume_dim>> /*meta*/) const {
    constexpr double default_initial_time = 0.;
    return {std::move(get<InitialDataPhi>(scalar_wave_data_.variables(
        x, default_initial_time, evolved_field_vars_tags{})))};
  }

  // Retrieve one or more tags
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, volume_dim>& x,
                                         tmpl::list<Tags...> /*meta*/) const {
    static_assert(sizeof...(Tags) > 1,
                  "This generic template will recurse infinitely if only one "
                  "tag is being retrieved through it.");

    static_assert(tmpl2::flat_all_v<tmpl::list_contains_v<tags, Tags>...>,
                  "At least one of the requested tags is not supported.");

    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  template <typename LocalScalarFieldData,                  // NOLINT
            typename LocalBackgroundData>                   // NOLINT
  friend bool                                               // NOLINT
  operator==(const ScalarWaveGr<LocalScalarFieldData,       // NOLINT
                                LocalBackgroundData>& lhs,  // NOLINT
             const ScalarWaveGr<LocalScalarFieldData,       // NOLINT
                                LocalBackgroundData>& rhs)  // NOLINT
      ;                                                     // NOLINT

  ScalarFieldData scalar_wave_data_;
  BackgroundGrData background_gr_data_;
};

template <typename ScalarFieldData, typename BackgroundData>
bool operator!=(const ScalarWaveGr<ScalarFieldData, BackgroundData>& lhs,
                const ScalarWaveGr<ScalarFieldData, BackgroundData>& rhs);

}  // namespace AnalyticData
}  // namespace CurvedScalarWave
