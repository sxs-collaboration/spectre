// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <pup.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Elasticity::ConstitutiveRelations {

/*!
 * \brief A cubic crystalline material
 *
 * \details For a cubic crystalline material the Elasticity tensor in the linear
 * constitutive relation \f$T^{ij}=-Y^{ijkl}S_{kl}\f$ reduces to
 *
 * \f[
 * Y^{ijkl} =
 * \begin{cases}
 * c_{11} & \mathrm{for}\; i=j=k=l \\
 * c_{12} & \mathrm{for}\; i=j,k=l,i \neq k \\
 * c_{44} & \mathrm{for}\; i=k,j=l,i \neq j \;\mathrm{or}\; i=l,j=k,i\neq j \\
 * \end{cases}
 * \f]
 *
 * with the three independent parameters: the \f$\mathrm{Lam\acute{e}}\f$
 * parameter \f$\lambda\f$, the Shear modulus \f$\mu\f$ and the %Poisson
 * ratio \f$\nu\f$. In the parametrization chosen in this implementation we use
 * the experimental group parameters \f$c_{11}\f$, \f$c_{12}\f$ and
 * \f$c_{44}\f$, related by;
 *
 * \f[
 * c_{11} = \frac{1 - \nu}{\nu} \lambda = \frac{(1 - \nu)E}{(1 + \nu)(1 -
 * 2\nu)}, \quad
 * c_{12} = \lambda = \frac{E\nu}{(1 + \nu)(1 - 2\nu)}, \quad
 * c_{44} = \mu
 * \f]
 *
 * and inversely;
 *
 * \f[
 * E = \frac{(c_{11} + 2c_{12})(c_{11} - c_{12})}{c_{11} + c_{12}}, \quad
 * \nu = \left(1 + \frac{c_{11}}{c_{12}}\right)^{-1}, \quad
 * \mu = c_{44}
 * \f]
 *
 * The stress-strain relation then reduces to
 *
 * \f[
 * T^{ij} =
 * \begin{cases}
 * -(c_{11} - c_{12}) S^{ij} - c_{12} \mathrm{Tr}(S) & \mathrm{for}\; i=j \\
 * -2 c_{44} S^{ij} & \mathrm{for}\; i \neq j \\
 * \end{cases}
 * \f]
 *
 * In the case where the shear modulus satisfies \f$c_{44} =
 * \frac{c_{11}-c_{12}}{2}\f$ the constitutive relation is that of an isotropic
 * material (see `Elasticity::ConstitutiveRelations::IsotropicHomogeneous`).
 */

class CubicCrystal : public ConstitutiveRelation<3> {
 public:
  static constexpr size_t volume_dim = 3;

  struct C_11 {
    using type = double;
    static constexpr Options::String help = {
        "c_11 parameter for a cubic crystal"};
    static type lower_bound() noexcept { return 0.0; }
  };

  struct C_12 {
    using type = double;
    static constexpr Options::String help = {
        "c_12 parameter for a cubic crystal"};
    static type lower_bound() noexcept { return 0.0; }
  };

  struct C_44 {
    using type = double;
    static constexpr Options::String help = {
        "c_44 parameter for a cubic crystal"};
    static type lower_bound() noexcept { return 0.0; }
  };

  using options = tmpl::list<C_11, C_12, C_44>;

  static constexpr Options::String help = {
      "A constitutive relation that describes a cubic, crystalline material in "
      "terms of the three independent group paremeters. The parameters "
      "are measured in units of stress, typically Pascals."};

  CubicCrystal() = default;
  CubicCrystal(const CubicCrystal&) = default;
  CubicCrystal& operator=(const CubicCrystal&) = default;
  CubicCrystal(CubicCrystal&&) = default;
  CubicCrystal& operator=(CubicCrystal&&) = default;
  ~CubicCrystal() override = default;

  CubicCrystal(double c_11, double c_12, double c_44) noexcept;

  /// The constitutive relation that characterizes the elastic properties of a
  /// material
  tnsr::II<DataVector, 3> stress(const tnsr::ii<DataVector, 3>& strain,
                                 const tnsr::I<DataVector, 3>& x) const
      noexcept override;

  /// The 1st group parameter \f$c_{11} = \frac{1 - \nu}{\nu} \lambda\f$
  double c_11() const noexcept;
  /// The 2nd group parameter; the \f$\mathrm{Lam\acute{e}}\f$ parameter
  /// \f$c_{12} = \lambda\f$
  double c_12() const noexcept;
  /// The 3rd group parameter; the shear modulus (rigidity) \f$c_{44} = \mu\f$
  double c_44() const noexcept;
  /// The Young's modulus \f$E\f$
  double youngs_modulus() const noexcept;
  /// The %Poisson ratio \f$\nu\f$
  double poisson_ratio() const noexcept;

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept override;  //  NOLINT

  explicit CubicCrystal(CkMigrateMessage* /*unused*/) noexcept {}

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(ConstitutiveRelation<3>), CubicCrystal);

 private:
  double c_11_ = std::numeric_limits<double>::signaling_NaN();
  double c_12_ = std::numeric_limits<double>::signaling_NaN();
  double c_44_ = std::numeric_limits<double>::signaling_NaN();
};  // namespace ConstitutiveRelations

bool operator==(const CubicCrystal& lhs, const CubicCrystal& rhs) noexcept;
bool operator!=(const CubicCrystal& lhs, const CubicCrystal& rhs) noexcept;

}  // namespace Elasticity::ConstitutiveRelations
