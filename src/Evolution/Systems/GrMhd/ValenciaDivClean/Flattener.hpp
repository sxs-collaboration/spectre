// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOptions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
namespace PUP {
class er;
}  // namespace PUP
template <size_t Dim>
class Mesh;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
template <typename TagsList>
class Variables;
/// \endcond

namespace grmhd::ValenciaDivClean {
/*!
 * \brief Reduces oscillations inside an element in an attempt to guarantee a
 * physical solution of the conserved variables for which the primitive
 * variables can be recovered.
 *
 * The algorithm uses the conditions of FixConservatives on \f$\tilde{D}\f$ and
 * \f$\tilde{\tau}\f$ to reduce oscillations inside an element. Oscillations are
 * reduced by rescaling the conserved variables about the mean to bring them
 * into the required range. When rescaling \f$\tilde{D}\f$ because it is
 * negative, it is important to also rescale \f$\tilde{\tau}\f$ and
 * \f$\tilde{S}_i\f$ by the same amount. At least, this is what is observed in
 * the cylindrical blast wave test problem.
 *
 * This currently doesn't use the check on \f$\tilde{S}^2\f$, but instead checks
 * that the primitive variables can be recovered. If the primitives cannot be
 * recovered then we flatten to the mean values in the element.
 */
template <typename RecoverySchemesList>
class Flattener {
 public:
  /// \brief Require that the mean of TildeD is positive, otherwise terminate
  /// the simulation.
  struct RequirePositiveMeanTildeD {
    using type = bool;
    static constexpr Options::String help = {
        "Require that the mean of TildeD is positive, otherwise terminate the "
        "simulation."};
  };

  /// \brief Require that the mean of TildeYe is positive, otherwise terminate
  /// the simulation.
  struct RequirePositiveMeanTildeYe {
    using type = bool;
    static constexpr Options::String help = {
        "Require that the mean of TildeYe is positive, otherwise terminate the "
        "simulation."};
  };

  /// \brief Require that the mean of TildeTau is physical, otherwise terminate
  /// the simulation.
  struct RequirePhysicalMeanTildeTau {
    using type = bool;
    static constexpr Options::String help = {
        "Require that the mean of TildeTau is physical, otherwise terminate "
        "the simulation."};
  };

  /// \brief If true, then the primitive variables are updated at the end of the
  /// function.
  ///
  /// This is useful when not using any pointwise conserved variable fixing,
  /// treating the case that the means do not satisfy the bounds as an error.
  struct RecoverPrimitives {
    using type = bool;
    static constexpr Options::String help = {
        "If true, then the primitive variables are updated at the end of the "
        "function."};
  };

  using options =
      tmpl::list<RequirePositiveMeanTildeD, RequirePositiveMeanTildeYe,
                 RequirePhysicalMeanTildeTau, RecoverPrimitives>;
  static constexpr Options::String help = {
      "Reduces oscillations (flattens) the conserved variables according to "
      "the variable fixing procedure described in Foucart's thesis.\n"};

  Flattener(bool require_positive_mean_tilde_d,
            bool require_positive_mean_tilde_ye,
            bool require_physical_mean_tilde_tau, bool recover_primitives);

  Flattener() = default;
  Flattener(const Flattener& /*rhs*/) = default;
  Flattener& operator=(const Flattener& /*rhs*/) = default;
  Flattener(Flattener&& /*rhs*/) = default;
  Flattener& operator=(Flattener&& /*rhs*/) = default;
  ~Flattener() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  using return_tags =
      tmpl::list<Tags::TildeD, Tags::TildeYe, Tags::TildeTau,
                 Tags::TildeS<Frame::Inertial>,
                 ::Tags::Variables<hydro::grmhd_tags<DataVector>>>;
  using argument_tags = tmpl::list<
      Tags::TildeB<>, Tags::TildePhi,
      gr::Tags::SqrtDetSpatialMetric<DataVector>,
      gr::Tags::SpatialMetric<DataVector, 3>,
      gr::Tags::InverseSpatialMetric<DataVector, 3>, domain::Tags::Mesh<3>,
      domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>,
      hydro::Tags::GrmhdEquationOfState,
      grmhd::ValenciaDivClean::Tags::PrimitiveFromConservativeOptions>;

  void operator()(
      gsl::not_null<Scalar<DataVector>*> tilde_d,
      gsl::not_null<Scalar<DataVector>*> tilde_ye,
      gsl::not_null<Scalar<DataVector>*> tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3>*> tilde_s,
      gsl::not_null<Variables<hydro::grmhd_tags<DataVector>>*> primitives,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
      const Scalar<DataVector>& tilde_phi,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
      const Mesh<3>& mesh,
      const Scalar<DataVector>& det_logical_to_inertial_inv_jacobian,
      const EquationsOfState::EquationOfState<true, 3>& eos,
      const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions&
          primitive_from_conservative_options) const;

 private:
  template <typename LocalRecoverySchemesList>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const Flattener<LocalRecoverySchemesList>& lhs,
                         const Flattener<LocalRecoverySchemesList>& rhs);

  bool require_positive_mean_tilde_d_ = false;
  bool require_positive_mean_tilde_ye_ = false;
  bool require_physical_mean_tilde_tau_ = false;
  bool recover_primitives_ = false;
};

template <typename RecoverySchemesList>
bool operator!=(const Flattener<RecoverySchemesList>& lhs,
                const Flattener<RecoverySchemesList>& rhs);
}  // namespace grmhd::ValenciaDivClean
