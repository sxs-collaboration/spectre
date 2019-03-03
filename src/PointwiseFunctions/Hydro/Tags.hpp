// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"

/// \ingroup EvolutionSystemsGroup
/// \brief Items related to hydrodynamic systems.
namespace hydro {
/// %Tags for hydrodynamic systems.
namespace Tags {

/// The Alfvén speed squared \f$v_A^2\f$.
template <typename DataType>
struct AlfvenSpeedSquared : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "AlfvenSpeedSquared"; }
};

/// The magnetic field \f$b^\mu = u_\nu {}^\star\!F^{\mu \nu}\f$
/// measured by an observer comoving with the fluid with 4-velocity
/// \f$u_\nu\f$ where \f${}^\star\!F^{\mu \nu}\f$
/// is the dual of the Faraday tensor.  Note that \f$b^\mu\f$ has a
/// time component (that is, \f$b^\mu n_\mu \neq 0\f$, where \f$n_\mu\f$ is
/// the normal to the spacelike hypersurface).
template <typename DataType, size_t Dim, typename Fr>
struct ComovingMagneticField : db::SimpleTag {
  using type = tnsr::A<DataType, Dim, Fr>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "ComovingMagneticField";
  }
};

/// The square of the comoving magnetic field, \f$b^\mu b_\mu\f$
template <typename DataType>
struct ComovingMagneticFieldSquared : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "ComovingMagneticFieldSquared"; }
};

/// The divergence-cleaning field \f$\Phi\f$.
template <typename DataType>
struct DivergenceCleaningField : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "DivergenceCleaningField"; }
};

/// Base tag for the equation of state
struct EquationOfStateBase : db::BaseTag {};

/// The equation of state
template <typename EquationOfStateType>
struct EquationOfState : EquationOfStateBase, db::SimpleTag {
  using type = EquationOfStateType;
  static std::string name() noexcept { return "EquationOfState"; }
};

/// The Lorentz factor \f$W = (1-v^iv_i)^{-1/2}\f$, where \f$v^i\f$ is
/// the spatial velocity of the fluid.
template <typename DataType>
struct LorentzFactor : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "LorentzFactor"; }
};

/// The square of the Lorentz factor \f$W^2\f$.
template <typename DataType>
struct LorentzFactorSquared {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "LorentzFactorSquared"; }
};

/// The magnetic field \f$B^i = n_\mu {}^\star\!F^{i \mu}\f$ measured by an
/// Eulerian observer, where \f$n_\mu\f$ is the normal to the spatial
/// hypersurface and \f${}^\star\!F^{\mu \nu}\f$ is the dual of the
/// Faraday tensor.  Note that \f$B^i\f$ is purely spatial, and it
/// can be lowered using the spatial metric.
template <typename DataType, size_t Dim, typename Fr>
struct MagneticField : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Fr>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "MagneticField";
  }
};

/// The magnetic field dotted into the spatial velocity, \f$B^iv_i\f$ where
/// \f$v_i\f$ is the spatial velocity one-form.
template <typename DataType>
struct MagneticFieldDotSpatialVelocity : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept {
    return "MagneticFieldDotSpatialVelocity";
  }
};

/// The one-form of the magnetic field.  Note that \f$B^i\f$ is raised
/// and lowered with the spatial metric.
/// \see hydro::Tags::MagneticField
template <typename DataType, size_t Dim, typename Fr>
struct MagneticFieldOneForm : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Fr>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "MagneticFieldOneForm";
  }
};

/// The square of the magnetic field, \f$B^iB_i\f$
template <typename DataType>
struct MagneticFieldSquared : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "MagneticFieldSquared"; }
};

/// The magnetic pressure \f$p_m\f$.
template <typename DataType>
struct MagneticPressure : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "MagneticPressure"; }
};

/// The fluid pressure \f$p\f$.
template <typename DataType>
struct Pressure : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "Pressure"; }
};

/// The rest-mass density \f$\rho\f$.
template <typename DataType>
struct RestMassDensity : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "RestMassDensity"; }
};

/// The sound speed squared \f$c_s^2\f$.
template <typename DataType>
struct SoundSpeedSquared : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "SoundSpeedSquared"; }
};

/// The spatial velocity \f$v^i\f$ of the fluid,
/// where \f$v^i=u^i/W + \beta^i/\alpha\f$.
/// Here \f$u^i\f$ is the spatial part of the 4-velocity of the fluid,
/// \f$W\f$ is the Lorentz factor, \f$\beta^i\f$ is the shift vector,
/// and \f$\alpha\f$ is the lapse function. Note that \f$v^i\f$ is raised
/// and lowered with the spatial metric.
template <typename DataType, size_t Dim, typename Fr>
struct SpatialVelocity : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Fr>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "SpatialVelocity";
  }
};

/// The spatial velocity one-form \f$v_i\f$, where \f$v_i\f$ is raised
/// and lowered with the spatial metric.
template <typename DataType, size_t Dim, typename Fr>
struct SpatialVelocityOneForm : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Fr>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "SpatialVelocityOneForm";
  }
};

/// The spatial velocity squared \f$v^2 = v_i v^i\f$.
template <typename DataType>
struct SpatialVelocitySquared : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "SpatialVelocitySquared"; }
};

/// The specific enthalpy \f$h\f$.
template <typename DataType>
struct SpecificEnthalpy : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "SpecificEnthalpy"; }
};

/// The specific internal energy \f$\epsilon\f$.
template <typename DataType>
struct SpecificInternalEnergy : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "SpecificInternalEnergy"; }
};

}  // namespace Tags

/// The tags for the primitive variables for GRMHD.
template <typename DataType>
using grmhd_tags =
    tmpl::list<hydro::Tags::RestMassDensity<DataType>,
               hydro::Tags::SpecificInternalEnergy<DataType>,
               hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>,
               hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>,
               hydro::Tags::DivergenceCleaningField<DataType>,
               hydro::Tags::LorentzFactor<DataType>,
               hydro::Tags::Pressure<DataType>,
               hydro::Tags::SpecificEnthalpy<DataType>>;
}  // namespace hydro
