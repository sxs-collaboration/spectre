// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/PunctureField.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeodesicAcceleration.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace CurvedScalarWave::AnalyticData {

/*!
 * \brief Analytic initial data for a scalar point charge in Kerr-Schild
 * coordinates. This assumes the charge is initially on a geodesic orbit.
 *
 * \details The initial data corresponds to the zeroth order puncture field
 * which is effectively the Lorentz-boosted solution of a scalar charge in flat
 * space.
 */

class ZerothOrderPuncture
    : public evolution::initial_data::InitialData,
      public MarkAsAnalyticData
#if defined(SPECTRE_USE_FINDUS)
    ,
      public virtual findus::serialize::SerializableDerived<
          ZerothOrderPuncture, evolution::initial_data::InitialData>
#endif
{
 public:
  struct ParticlePosition {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "The initial position of the scalar charge."};
  };

  struct ParticleVelocity {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "The initial velocity of the scalar charge"};
  };

  struct ParticleCharge {
    using type = double;
    static constexpr Options::String help = {
        "The value of the particle's charge."};
    static constexpr double lower_bound() { return 0.; }
    static constexpr double upper_bound() { return 1.; }
  };

  using options =
      tmpl::list<ParticlePosition, ParticleVelocity, ParticleCharge>;

  static constexpr Options::String help = {
      "Initial data for a scalar charge in Kerr-Schild coordinates. It "
      "corresponds to the zeroth order puncture field which is the "
      "Lorentz-boosted solution of a scalar charge in flat space."};

  ZerothOrderPuncture() = default;

  ZerothOrderPuncture(std::array<double, 3> particle_position,
                      std::array<double, 3> particle_velocity,
                      double particle_charge,
                      const Options::Context& context = {});
  ZerothOrderPuncture(const ZerothOrderPuncture&) = default;
  ZerothOrderPuncture& operator=(const ZerothOrderPuncture&) = default;
  ZerothOrderPuncture(ZerothOrderPuncture&&) = default;
  ZerothOrderPuncture& operator=(ZerothOrderPuncture&&) = default;
  ~ZerothOrderPuncture() override = default;

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  WRAPPED_PUPable_decl_template(ZerothOrderPuncture);
  /// \endcond

  static constexpr size_t volume_dim = 3;
  using tags =
      tmpl::list<CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi,
                 CurvedScalarWave::Tags::Phi<3>>;

  /// Retrieve the evolution variables at spatial coordinates `x`
  tuples::TaggedTuple<CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi,
                      CurvedScalarWave::Tags::Phi<3>>
  variables(const tnsr::I<DataVector, 3>& x, tags /*meta*/) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) override;

 private:
  // assume a non-spinning black hole of mass 1M centered on the coordinate
  // origin
  gr::Solutions::KerrSchild kerr_schild_{1., {{0., 0., 0.}}, {{0., 0., 0.}}};
  tnsr::I<double, 3> particle_position_{
      std::numeric_limits<double>::signaling_NaN()};
  tnsr::I<double, 3> particle_velocity_{
      std::numeric_limits<double>::signaling_NaN()};
  tnsr::I<double, 3> geodesic_acceleration_{
      std::numeric_limits<double>::signaling_NaN()};
  double particle_charge_{std::numeric_limits<double>::signaling_NaN()};

  friend bool operator==(const ZerothOrderPuncture& lhs,
                         const ZerothOrderPuncture& rhs);

  friend bool operator!=(const ZerothOrderPuncture& lhs,
                         const ZerothOrderPuncture& rhs);
};

}  // namespace CurvedScalarWave::AnalyticData
