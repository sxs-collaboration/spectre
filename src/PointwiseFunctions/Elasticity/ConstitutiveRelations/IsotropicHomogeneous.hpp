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

namespace Elasticity {
namespace ConstitutiveRelations {

/*!
 * \brief An isotropic and homogeneous material
 *
 * \details For an isotropic and homogeneous material the linear constitutive
 * relation \f$T^{ij}=-Y^{ijkl}S_{kl}\f$ reduces to \f[
 * Y^{ijkl} = \lambda \delta^{ij}\delta^{kl} + \mu
 * \left(\delta^{ik}\delta^{jl} + \delta^{il}\delta^{jk}\right) \\
 * \implies \quad T^{ij} = -\lambda \mathrm{Tr}(S) \delta^{ij} - 2\mu S^{ij}
 * \f] with the _Lamé parameter_ \f$\lambda\f$ and the _shear modulus_ (or
 * _rigidity_) \f$\mu\f$. In the parametrization chosen in this implementation
 * we use the _bulk modulus_ (or _incompressibility_) \f[
 * K=\lambda + \frac{2}{3}\mu
 * \f] instead of the Lamé parameter. In this parametrization the
 * stress-strain relation \f[
 * T^{ij} = -K \mathrm{Tr}(S) \delta^{ij} - 2\mu\left(S^{ij} -
 * \frac{1}{3}\mathrm{Tr}(S)\delta^{ij}\right) \f]
 * decomposes into a scalar and a traceless part (Eq. 11.18 in
 * \cite ThorneBlandford2017). Parameters also often used in this context are
 * the _Young's modulus_ \f[
 * E=\frac{9K\mu}{3K+\mu}=\frac{\mu(3\lambda+2\mu)}{\lambda+\mu}
 * \f] and the _Poisson ratio_ \f[
 * \nu=\frac{3K-2\mu}{2(3K+\mu)}=\frac{\lambda}{2(\lambda+\mu)}
 * \f]. Inversely, these relations read: \f[
 * K &=\frac{E}{3(1-2\nu)} \\
 * \lambda &=\frac{E\nu}{(1+\nu)(1-2\nu)} \\
 * \mu &=\frac{E}{2(1+\nu)}
 * \f]
 *
 * **In two dimensions** this implementation reduces to the plane-stress
 * approximation. We assume that all stresses apply in the plane of the
 * computational domain, which corresponds to scenarios of in-plane stretching
 * and shearing of thin slabs of material. Since orthogonal stresses vanish as
 * \f$T^{i3}=0=T^{3i}\f$ we find \f$\mathrm{Tr}(S)=\frac{2\mu}{\lambda +
 * 2\mu}\mathrm{Tr}^{(2)}(S)\f$, where \f$\mathrm{Tr}^{(2)}\f$ denotes that the
 * trace only applies to the two dimensions within the plane. The constitutive
 * relation thus reduces to \f[
 * T^{ij}&=-\frac{2\lambda\mu}{\lambda + 2\mu}\mathrm{Tr}^{(2)}\delta^{ij} -
 * 2\mu S^{ij} \\
 * &=-\frac{E\nu}{1-\nu^2}\mathrm{Tr}^{(2)}\delta^{ij} - \frac{E}{1+\nu}S^{ij}
 * \f] which is non-zero only in the directions of the plane. Since the stresses
 * are also assumed to be constant along the thickness of the plane
 * \f$\partial_3T^{ij}=0\f$ the elasticity problem \f$-\partial_i T^{ij}=F^j\f$
 * reduces to two dimensions.
 */
template <size_t Dim>
class IsotropicHomogeneous : public ConstitutiveRelation<Dim> {
 public:
  struct BulkModulus {
    using type = double;
    static constexpr OptionString help = {
        "The incompressibility of the material"};
    static type lower_bound() noexcept { return 0.0; }
  };

  struct ShearModulus {
    using type = double;
    static constexpr OptionString help = {"The rigidity of the material"};
    static type lower_bound() noexcept { return 0.0; }
  };

  using options = tmpl::list<BulkModulus, ShearModulus>;

  static constexpr OptionString help = {
      "A constitutive relation that describes an isotropic, homogeneous "
      "material."};

  IsotropicHomogeneous() = default;
  IsotropicHomogeneous(const IsotropicHomogeneous&) = delete;
  IsotropicHomogeneous& operator=(const IsotropicHomogeneous&) = delete;
  IsotropicHomogeneous(IsotropicHomogeneous&&) = default;
  IsotropicHomogeneous& operator=(IsotropicHomogeneous&&) = default;
  ~IsotropicHomogeneous() override = default;

  IsotropicHomogeneous(double bulk_modulus, double shear_modulus) noexcept;

  /// The constitutive relation that characterizes the elastic properties of a
  /// material
  tnsr::II<DataVector, Dim> stress(const tnsr::ii<DataVector, Dim>& strain,
                                   const tnsr::I<DataVector, Dim>& x) const
      noexcept override;

  /// The bulk modulus (or incompressibility) \f$K\f$
  double bulk_modulus() const noexcept;
  /// The shear modulus (or rigidity) \f$\mu\f$
  double shear_modulus() const noexcept;
  /// The Lamé parameter \f$\lambda\f$
  double lame_parameter() const noexcept;
  /// The Young's modulus \f$E\f$
  double youngs_modulus() const noexcept;
  /// The Poisson ratio \f$\nu\f$
  double poisson_ratio() const noexcept;

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept override;  //  NOLINT

  explicit IsotropicHomogeneous(CkMigrateMessage* /*unused*/) noexcept {}

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(ConstitutiveRelation<Dim>), IsotropicHomogeneous);

 private:
  double bulk_modulus_ = std::numeric_limits<double>::signaling_NaN();
  double shear_modulus_ = std::numeric_limits<double>::signaling_NaN();
};

template <size_t Dim>
bool operator==(const IsotropicHomogeneous<Dim>& lhs,
                const IsotropicHomogeneous<Dim>& rhs) noexcept;
template <size_t Dim>
bool operator!=(const IsotropicHomogeneous<Dim>& lhs,
                const IsotropicHomogeneous<Dim>& rhs) noexcept;

/// \cond
template <size_t Dim>
PUP::able::PUP_ID IsotropicHomogeneous<Dim>::my_PUP_ID = 0;
/// \endcond

}  // namespace ConstitutiveRelations
}  // namespace Elasticity
