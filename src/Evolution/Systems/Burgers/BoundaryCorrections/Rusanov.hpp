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
 * \brief A Rusanov/local Lax-Friedrichs Riemann solver
 *
 * Let \f$U\f$ be the evolved variable, \f$F^i\f$ the flux, and \f$n_i\f$ be
 * the outward directed unit normal to the interface. Denoting \f$F := n_i
 * F^i\f$, the %Rusanov boundary correction is
 *
 * \f{align*}
 * G_\text{Rusanov} = \frac{F_\text{int} - F_\text{ext}}{2} -
 * \frac{\text{max}\left(|U_\text{int}|, |U_\text{ext}|\right)}{2}
 * \left(U_\text{ext} - U_\text{int}\right),
 * \f}
 *
 * where "int" and "ext" stand for interior and exterior. The minus sign in
 * front of the \f$F_{\text{ext}}\f$ is necessary because the outward directed
 * normal of the neighboring element has the opposite sign, i.e.
 * \f$n_i^{\text{ext}}=-n_i^{\text{int}}\f$.
 *
 * \note In the strong form the `dg_boundary_terms` function returns
 * \f$G - F_\text{int}\f$
 */
class Rusanov final : public BoundaryCorrection {
 private:
  struct AbsCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "Computes the Rusanov or local Lax-Friedrichs boundary correction term "
      "for the Burgers system."};

  Rusanov() = default;
  Rusanov(const Rusanov&) = default;
  Rusanov& operator=(const Rusanov&) = default;
  Rusanov(Rusanov&&) = default;
  Rusanov& operator=(Rusanov&&) = default;
  ~Rusanov() override = default;

  /// \cond
  explicit Rusanov(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Rusanov);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection> get_clone() const override;

  using dg_package_field_tags =
      tmpl::list<Tags::U, ::Tags::NormalDotFlux<Tags::U>, AbsCharSpeed>;
  using dg_package_data_temporary_tags = tmpl::list<>;
  using dg_package_data_volume_tags = tmpl::list<>;

  static double dg_package_data(
      gsl::not_null<Scalar<DataVector>*> packaged_u,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux,
      gsl::not_null<Scalar<DataVector>*> packaged_abs_char_speed,
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
