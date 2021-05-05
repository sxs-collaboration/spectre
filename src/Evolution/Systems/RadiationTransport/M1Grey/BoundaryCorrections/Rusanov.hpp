// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
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

namespace RadiationTransport::M1Grey::BoundaryCorrections {

namespace Rusanov_detail {
void dg_package_data_impl(
    gsl::not_null<Scalar<DataVector>*> packaged_tilde_e,
    gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> packaged_tilde_s,
    gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_e,
    gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_s,
    const Scalar<DataVector>& tilde_e,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_e,
    const tnsr::Ij<DataVector, 3, Frame::Inertial>& flux_tilde_s,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>& mesh_velocity,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity) noexcept;

void dg_boundary_terms_impl(
    gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_e,
    gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_s,
    const Scalar<DataVector>& tilde_e_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_e_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_s_int,
    const Scalar<DataVector>& tilde_e_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_e_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_s_ext,
    dg::Formulation dg_formulation) noexcept;
}  // namespace Rusanov_detail

/// \cond
template <typename NeutrinoSpeciesList>
class Rusanov;
/// \endcond

/*!
 * \brief A Rusanov/local Lax-Friedrichs Riemann solver
 *
 * Let \f$U\f$ be the state vector of evolved variables, \f$F^i\f$ the
 * corresponding fluxes, and \f$n_i\f$ be the outward directed unit normal to
 * the interface. Denoting \f$F := n_i F^i\f$, the %Rusanov boundary correction
 * is
 *
 * \f{align*}
 * G_\text{Rusanov} = \frac{F_\text{int} - F_\text{ext}}{2} -
 * \frac{\text{max}\left(\{|\lambda_\text{int}|\},
 * \{|\lambda_\text{ext}|\}\right)}{2} \left(U_\text{ext} - U_\text{int}\right),
 * \f}
 *
 * where "int" and "ext" stand for interior and exterior, and
 * \f$\{|\lambda|\}\f$ is the set of characteristic/signal speeds. The minus
 * sign in front of the \f$F_{\text{ext}}\f$ is necessary because the outward
 * directed normal of the neighboring element has the opposite sign, i.e.
 * \f$n_i^{\text{ext}}=-n_i^{\text{int}}\f$.
 *
 * For radiation, the max characteristic/signal speed is 1 (the speed of light).
 * Note that the characteristic speeds of this system are *not* yet fully
 * implemented, see the warning in the documentation for the characteristics.
 *
 * \note In the strong form the `dg_boundary_terms` function returns
 * \f$G - F_\text{int}\f$
 */
template <typename... NeutrinoSpecies>
class Rusanov<tmpl::list<NeutrinoSpecies...>> final
    : public BoundaryCorrection<tmpl::list<NeutrinoSpecies...>> {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "Computes the Rusanov or local Lax-Friedrichs boundary correction term "
      "for the M1Grey radiation transport system."};

  Rusanov() = default;
  Rusanov(const Rusanov&) = default;
  Rusanov& operator=(const Rusanov&) = default;
  Rusanov(Rusanov&&) = default;
  Rusanov& operator=(Rusanov&&) = default;
  ~Rusanov() override = default;

  /// \cond
  explicit Rusanov(CkMigrateMessage* msg) noexcept
      : BoundaryCorrection<tmpl::list<NeutrinoSpecies...>>(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Rusanov);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override {  // NOLINT
    BoundaryCorrection<tmpl::list<NeutrinoSpecies...>>::pup(p);
  }

  std::unique_ptr<BoundaryCorrection<tmpl::list<NeutrinoSpecies...>>>
  get_clone() const noexcept override {
    return std::make_unique<Rusanov>(*this);
  }

  using dg_package_field_tags = tmpl::list<
      Tags::TildeE<Frame::Inertial, NeutrinoSpecies>...,
      Tags::TildeS<Frame::Inertial, NeutrinoSpecies>...,
      ::Tags::NormalDotFlux<Tags::TildeE<Frame::Inertial, NeutrinoSpecies>>...,
      ::Tags::NormalDotFlux<Tags::TildeS<Frame::Inertial, NeutrinoSpecies>>...>;
  using dg_package_data_temporary_tags = tmpl::list<>;
  using dg_package_data_primitive_tags = tmpl::list<>;
  using dg_package_data_volume_tags = tmpl::list<>;

  double dg_package_data(
      const gsl::not_null<typename Tags::TildeE<
          Frame::Inertial, NeutrinoSpecies>::type*>... packaged_tilde_e,
      const gsl::not_null<typename Tags::TildeS<
          Frame::Inertial, NeutrinoSpecies>::type*>... packaged_tilde_s,
      const gsl::not_null<typename Tags::TildeE<
          Frame::Inertial,
          NeutrinoSpecies>::type*>... packaged_normal_dot_flux_tilde_e,
      const gsl::not_null<typename Tags::TildeS<
          Frame::Inertial,
          NeutrinoSpecies>::type*>... packaged_normal_dot_flux_tilde_s,

      const typename Tags::TildeE<Frame::Inertial,
                                  NeutrinoSpecies>::type&... tilde_e,
      const typename Tags::TildeS<Frame::Inertial,
                                  NeutrinoSpecies>::type&... tilde_s,
      const typename ::Tags::Flux<
          Tags::TildeE<Frame::Inertial, NeutrinoSpecies>, tmpl::size_t<3>,
          Frame::Inertial>::type&... flux_tilde_e,
      const typename ::Tags::Flux<
          Tags::TildeS<Frame::Inertial, NeutrinoSpecies>, tmpl::size_t<3>,
          Frame::Inertial>::type&... flux_tilde_s,

      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity)
      const noexcept {
    EXPAND_PACK_LEFT_TO_RIGHT(Rusanov_detail::dg_package_data_impl(
        packaged_tilde_e, packaged_tilde_s, packaged_normal_dot_flux_tilde_e,
        packaged_normal_dot_flux_tilde_s, tilde_e, tilde_s, flux_tilde_e,
        flux_tilde_s, normal_covector, normal_vector, mesh_velocity,
        normal_dot_mesh_velocity));
    // max speed = speed of light
    return 1.0;
  }

  void dg_boundary_terms(
      const gsl::not_null<typename Tags::TildeE<
          Frame::Inertial,
          NeutrinoSpecies>::type*>... boundary_correction_tilde_e,
      const gsl::not_null<typename Tags::TildeS<
          Frame::Inertial,
          NeutrinoSpecies>::type*>... boundary_correction_tilde_s,
      const typename Tags::TildeE<Frame::Inertial,
                                  NeutrinoSpecies>::type&... tilde_e_int,
      const typename Tags::TildeS<Frame::Inertial,
                                  NeutrinoSpecies>::type&... tilde_s_int,
      const typename Tags::TildeE<Frame::Inertial, NeutrinoSpecies>::
          type&... normal_dot_flux_tilde_e_int,
      const typename Tags::TildeS<Frame::Inertial, NeutrinoSpecies>::
          type&... normal_dot_flux_tilde_s_int,
      const typename Tags::TildeE<Frame::Inertial,
                                  NeutrinoSpecies>::type&... tilde_e_ext,
      const typename Tags::TildeS<Frame::Inertial,
                                  NeutrinoSpecies>::type&... tilde_s_ext,
      const typename Tags::TildeE<Frame::Inertial, NeutrinoSpecies>::
          type&... normal_dot_flux_tilde_e_ext,
      const typename Tags::TildeS<Frame::Inertial, NeutrinoSpecies>::
          type&... normal_dot_flux_tilde_s_ext,
      dg::Formulation dg_formulation) const noexcept {
    EXPAND_PACK_LEFT_TO_RIGHT(Rusanov_detail::dg_boundary_terms_impl(
        boundary_correction_tilde_e, boundary_correction_tilde_s, tilde_e_int,
        tilde_s_int, normal_dot_flux_tilde_e_int, normal_dot_flux_tilde_s_int,
        tilde_e_ext, tilde_s_ext, normal_dot_flux_tilde_e_ext,
        normal_dot_flux_tilde_s_ext, dg_formulation));
  }
};

/// \cond
template <typename... NeutrinoSpecies>
// NOLINTNEXTLINE
PUP::able::PUP_ID Rusanov<tmpl::list<NeutrinoSpecies...>>::my_PUP_ID = 0;
/// \endcond

}  // namespace RadiationTransport::M1Grey::BoundaryCorrections
