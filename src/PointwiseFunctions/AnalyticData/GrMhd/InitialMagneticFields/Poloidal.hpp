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

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

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
 * Note that the coordinates are relative to the `Center` passed in, so the
 * field can be centered about any arbitrary point. The field is also zero
 * outside of `MaxDistanceFromCenter`, so that compact support can be imposed if
 * necessary.
 *
 * \warning This assumes the magnetic field is initialized, both in size and
 * value, before being passed into the `variables` function. This is so that
 * multiple magnetic fields can be superposed. Each magnetic field
 * configuration does a `+=` to make this possible.
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

  struct Center {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "The center of the magnetic field."};
  };

  struct MaxDistanceFromCenter {
    using type = double;
    static constexpr Options::String help = {
        "The maximum distance from the center to compute the magnetic field. "
        "Everywhere outside the field is set to zero."};
    static type lower_bound() { return 0.0; }
  };

  using options =
      tmpl::list<PressureExponent, CutoffPressure, VectorPotentialAmplitude,
                 Center, MaxDistanceFromCenter>;

  static constexpr Options::String help = {"Poloidal initial magnetic field"};

  Poloidal() = default;
  Poloidal(const Poloidal& /*rhs*/) = default;
  Poloidal& operator=(const Poloidal& /*rhs*/) = default;
  Poloidal(Poloidal&& /*rhs*/) = default;
  Poloidal& operator=(Poloidal&& /*rhs*/) = default;
  ~Poloidal() override = default;

  Poloidal(size_t pressure_exponent, double cutoff_pressure,
           double vector_potential_amplitude, std::array<double, 3> center,
           double max_distance_from_center);

  auto get_clone() const -> std::unique_ptr<InitialMagneticField> override;

  /// \cond
  explicit Poloidal(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Poloidal);
  /// \endcond

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  /// Retrieve magnetic fields at `(x)`
  void variables(gsl::not_null<tnsr::I<DataVector, 3>*> result,
                 const tnsr::I<DataVector, 3>& coords,
                 const Scalar<DataVector>& pressure,
                 const Scalar<DataVector>& sqrt_det_spatial_metric,
                 const tnsr::i<DataVector, 3>& deriv_pressure) const override;

  /// Retrieve magnetic fields at `(x)`
  void variables(gsl::not_null<tnsr::I<double, 3>*> result,
                 const tnsr::I<double, 3>& coords,
                 const Scalar<double>& pressure,
                 const Scalar<double>& sqrt_det_spatial_metric,
                 const tnsr::i<double, 3>& deriv_pressure) const override;

  bool is_equal(const InitialMagneticField& rhs) const override;

 private:
  template <typename DataType>
  void variables_impl(gsl::not_null<tnsr::I<DataType, 3>*> magnetic_field,
                      const tnsr::I<DataType, 3>& coords,
                      const Scalar<DataType>& pressure,
                      const Scalar<DataType>& sqrt_det_spatial_metric,
                      const tnsr::i<DataType, 3>& deriv_pressure) const;

  size_t pressure_exponent_ = std::numeric_limits<size_t>::max();
  double cutoff_pressure_ = std::numeric_limits<double>::signaling_NaN();
  double vector_potential_amplitude_ =
      std::numeric_limits<double>::signaling_NaN();
  std::array<double, 3> center_{{std::numeric_limits<double>::signaling_NaN(),
                                 std::numeric_limits<double>::signaling_NaN(),
                                 std::numeric_limits<double>::signaling_NaN()}};
  double max_distance_from_center_ =
      std::numeric_limits<double>::signaling_NaN();

  friend bool operator==(const Poloidal& lhs, const Poloidal& rhs);
  friend bool operator!=(const Poloidal& lhs, const Poloidal& rhs);
};

}  // namespace grmhd::AnalyticData::InitialMagneticFields
