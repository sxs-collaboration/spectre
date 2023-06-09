// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace gh::gauges {
/*!
 * \brief Imposes the harmonic gauge condition, \f$H_a=0\f$.
 */
class Harmonic final : public GaugeCondition {
 public:
  using options = tmpl::list<>;

  static constexpr Options::String help{
      "Apply the Harmonic gauge condition H_a=0."};

  Harmonic() = default;
  Harmonic(const Harmonic&) = default;
  Harmonic& operator=(const Harmonic&) = default;
  Harmonic(Harmonic&&) = default;
  Harmonic& operator=(Harmonic&&) = default;
  ~Harmonic() override = default;

  /// \cond
  explicit Harmonic(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Harmonic);  // NOLINT
  /// \endcond

  template <size_t SpatialDim>
  void gauge_and_spacetime_derivative(
      gsl::not_null<tnsr::a<DataVector, SpatialDim, Frame::Inertial>*> gauge_h,
      gsl::not_null<tnsr::ab<DataVector, SpatialDim, Frame::Inertial>*>
          d4_gauge_h,
      double time,
      const tnsr::I<DataVector, SpatialDim, Frame::Inertial>& inertial_coords)
      const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  std::unique_ptr<GaugeCondition> get_clone() const override;
};
}  // namespace gh::gauges
