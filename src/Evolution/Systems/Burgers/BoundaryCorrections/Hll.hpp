// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Burgers::BoundaryCorrections {
/*!
 * \brief An HLL (Harten-Lax-van Leer) Riemann solver
 *
 * Let \f$U\f$ be the evolved variable, \f$F^i\f$ the flux, and \f$n_i\f$ be
 * the outward directed unit normal to the interface. Denoting \f$F := n_i
 * F^i\f$, the HLL boundary correction is \cite Harten1983
 *
 * \f{align*}
 * G_\text{HLL} = \frac{\lambda_\text{max} F_\text{int} +
 * \lambda_\text{min} F_\text{ext}}{\lambda_\text{max} - \lambda_\text{min}}
 * - \frac{\lambda_\text{min}\lambda_\text{max}}{\lambda_\text{max} -
 * \lambda_\text{min}} \left(U_\text{int} - U_\text{ext}\right)
 * \f}
 *
 * where "int" and "ext" stand for interior and exterior, and \cite Davis1988
 *
 * \f{align*}
 * \lambda_\text{min} &= \text{min}\left(U_\text{int},-U_\text{ext}, 0\right) \\
 * \lambda_\text{max} &= \text{max}\left(U_\text{int},-U_\text{ext}, 0\right),
 * \f}
 *
 * The positive sign in front of the \f$F_{\text{ext}}\f$ in \f$G_\text{HLL}\f$
 * and the minus signs in front of the \f$U_\text{ext}\f$ terms in
 * \f$\lambda_\text{min}\f$ and \f$\lambda_\text{max}\f$ terms are necessary
 * because the outward directed normal of the neighboring element has the
 * opposite sign, i.e. \f$n_i^{\text{ext}}=-n_i^{\text{int}}\f$. Note that for
 * either \f$\lambda_\text{min} = 0\f$ or \f$\lambda_\text{max} = 0\f$ (i.e. all
 * characteristics move in the same direction) the HLL boundary correction
 * reduces to pure upwinding.
 *
 * \note
 * - In the strong form the `dg_boundary_terms` function returns
 * \f$G - F_\text{int}\f$
 * - Some references use \f$S\f$ instead of \f$\lambda\f$ for the
 * signal/characteristic speeds
 */
class Hll final : public BoundaryCorrection {
 private:
  struct CharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "Computes the HLL boundary correction term for the Burgers system."};

  Hll() = default;
  Hll(const Hll&) = default;
  Hll& operator=(const Hll&) = default;
  Hll(Hll&&) = default;
  Hll& operator=(Hll&&) = default;
  ~Hll() override = default;

  /// \cond
  explicit Hll(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Hll);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection> get_clone() const override;

  using dg_package_field_tags =
      tmpl::list<Tags::U, ::Tags::NormalDotFlux<Tags::U>, CharSpeed>;
  using dg_package_data_temporary_tags = tmpl::list<>;
  using dg_package_data_volume_tags = tmpl::list<>;

  static double dg_package_data(
      gsl::not_null<Scalar<DataVector>*> packaged_u,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux,
      gsl::not_null<Scalar<DataVector>*> packaged_char_speed,
      const Scalar<DataVector>& u,
      const tnsr::I<DataVector, 1, Frame::Inertial>& flux_u,
      const tnsr::i<DataVector, 1, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, 1, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity);

  static void dg_boundary_terms(
      gsl::not_null<Scalar<DataVector>*> boundary_correction_u,
      const Scalar<DataVector>& u_int,
      const Scalar<DataVector>& normal_dot_flux_u_int,
      const Scalar<DataVector>& abs_char_speed_int,
      const Scalar<DataVector>& u_ext,
      const Scalar<DataVector>& normal_dot_flux_u_ext,
      const Scalar<DataVector>& abs_char_speed_ext,
      dg::Formulation dg_formulation);
};
}  // namespace Burgers::BoundaryCorrections
