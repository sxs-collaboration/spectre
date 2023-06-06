// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/InitialMagneticFields/InitialMagneticField.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace grmhd::AnalyticData::InitialMagneticFields {

/*!
 * \brief %Poloidal magnetic field for GRMHD initial data.
 *
 * The vector potential has the form
 *
 * \f{align*}{
 *  A_{\phi} = A_b \varpi^2 \max(p-p_{\mathrm{cut}}, 0)^{n_s} ,
 * \f}
 *
 * where \f$A_b\f$ controls the amplitude of the magnetic field,
 * \f$\varpi^2=x^2+y^2=r^2-z^2\f$ is the cylindrical radius,
 * \f$n_s\f$ controls the degree of differentiability, and
 * \f$p_{\mathrm{cut}}\f$ controls the pressure cutoff below which the magnetic
 * field is zero.
 *
 * In Cartesian coordinates the vector potential is:
 *
 * \f{align*}{
 *   A_x & = -\frac{y}{\varpi^2}A_\phi
 *            = -y A_b\max(p-p_{\mathrm{cut}}, 0)^{n_s}, \\
 *   A_y & = \frac{x}{\varpi^2}A_\phi
 *            = x A_b\max(p-p_{\mathrm{cut}}, 0)^{n_s}, \\
 *   A_z & = 0,
 * \f}
 *
 * On the region where the field is non-zero, the magnetic field is given by:
 *
 * \f{align*}{
 *   B^x & = -\frac{A_b n_s}{\sqrt{\gamma}}
 *        (p-p_{\mathrm{cut}})^{n_s-1} x \partial_z p \\
 *   B^x & = -\frac{A_b n_s}{\sqrt{\gamma}}
 *        (p-p_{\mathrm{cut}})^{n_s-1} y \partial_z p \\
 *   B^z & = \frac{A_b}{\sqrt{\gamma}}\left[
 *        2(p-p_{\mathrm{cut}})^{n_s}
 *        + n_s (p-p_{\mathrm{cut}})^{n_s-1} (x \partial_x p + y \partial_y p)
 *        \right]
 * \f}
 *
 * Taking the small-\f$r\f$ limit gives the magnetic field at the origin:
 *
 * \f{align*}{
 *   B^x&=0, \\
 *   B^y&=0, \\
 *   B^z&=\frac{A_b}{\sqrt{\gamma}}
 *        2(p-p_{\mathrm{cut}})^{n_s}.
 * \f}
 *
 */
class Poloidal : public InitialMagneticField {
 public:
  struct PressureExponent {
    using type = size_t;
    static constexpr Options::String help = {
        "The exponent n_s controlling the smoothness of the field"};
  };

  struct CutoffPressure {
    using type = double;
    static constexpr Options::String help = {
        "The pressure below which there is no magnetic field."};
    static type lower_bound() { return 0.0; }
  };

  struct VectorPotentialAmplitude {
    using type = double;
    static constexpr Options::String help = {
        "The amplitude A_b of the phi-component of the vector potential. This "
        "controls the magnetic field strength."};
    static type lower_bound() { return 0.0; }
  };

  using options =
      tmpl::list<PressureExponent, CutoffPressure, VectorPotentialAmplitude>;

  static constexpr Options::String help = {"Poloidal initial magnetic field"};

  Poloidal() = default;
  Poloidal(const Poloidal& /*rhs*/) = default;
  Poloidal& operator=(const Poloidal& /*rhs*/) = default;
  Poloidal(Poloidal&& /*rhs*/) = default;
  Poloidal& operator=(Poloidal&& /*rhs*/) = default;
  ~Poloidal() override = default;

  Poloidal(size_t pressure_exponent, double cutoff_pressure,
           double vector_potential_amplitude);

  auto get_clone() const -> std::unique_ptr<InitialMagneticField> override;

  /// \cond
  explicit Poloidal(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Poloidal);
  /// \endcond

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  /// Retrieve magnetic fields at `(x)`
  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& coords,
                 const Scalar<DataType>& pressure,
                 const Scalar<DataType>& sqrt_det_spatial_metric,
                 const tnsr::i<DataType, 3>& dcoords_pressure) const
      -> tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>;

 private:
  size_t pressure_exponent_ = std::numeric_limits<size_t>::max();
  double cutoff_pressure_ = std::numeric_limits<double>::signaling_NaN();
  double vector_potential_amplitude_ =
      std::numeric_limits<double>::signaling_NaN();

  friend bool operator==(const Poloidal& lhs, const Poloidal& rhs);
  friend bool operator!=(const Poloidal& lhs, const Poloidal& rhs);
};

}  // namespace grmhd::AnalyticData::InitialMagneticFields
