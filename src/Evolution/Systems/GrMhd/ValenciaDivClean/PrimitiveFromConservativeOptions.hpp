// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace grmhd::ValenciaDivClean {

/// Options to be passed to the Con2Prim algorithm.
/// Currently, we simply set a threshold for tildeD
/// below which the inversion is not performed and
/// the density is set to atmosphere values.
class PrimitiveFromConservativeOptions {
 public:
  struct CutoffDForInversion {
    static std::string name() { return "CutoffDForInversion"; }
    static constexpr Options::String help{
        "Value of density times Lorentz factor below which we skip "
        "conservative to primitive inversion."};
    using type = double;
    static type lower_bound() { return 0.0; }
  };

  struct DensityWhenSkippingInversion {
    static std::string name() { return "DensityWhenSkippingInversion"; }
    static constexpr Options::String help{
        "Value of density when we skip conservative to primitive inversion."};
    using type = double;
    static type lower_bound() { return 0.0; }
  };

  struct KastaunMaxLorentzFactor {
    static constexpr Options::String help{
        "The maximum Lorentz allowed during primitive recovery when using the "
        "Kastaun schemes."};
    using type = double;
    static type lower_bound() { return 1.0; }
  };

  using options = tmpl::list<CutoffDForInversion, DensityWhenSkippingInversion,
                             KastaunMaxLorentzFactor>;

  static constexpr Options::String help{
      "Options given to conservative to primitive inversion."};

  PrimitiveFromConservativeOptions() = default;

  PrimitiveFromConservativeOptions(double cutoff_d_for_inversion,
                                   double density_when_skipping_inversion,
                                   double kastaun_max_lorentz_factor);

  void pup(PUP::er& p);

  double cutoff_d_for_inversion() const { return cutoff_d_for_inversion_; }
  double density_when_skipping_inversion() const {
    return density_when_skipping_inversion_;
  }
  double kastaun_max_lorentz_factor() const {
    return kastaun_max_lorentz_factor_;
  }

 private:
  friend bool operator==(const PrimitiveFromConservativeOptions& lhs,
                         const PrimitiveFromConservativeOptions& rhs);

  double cutoff_d_for_inversion_ = std::numeric_limits<double>::signaling_NaN();
  double density_when_skipping_inversion_ =
      std::numeric_limits<double>::signaling_NaN();
  double kastaun_max_lorentz_factor_ =
      std::numeric_limits<double>::signaling_NaN();
};

bool operator!=(const PrimitiveFromConservativeOptions& lhs,
                const PrimitiveFromConservativeOptions& rhs);

}  // namespace grmhd::ValenciaDivClean
