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

/// The magnetic field \f$b^\mu = u_\nu F^{\mu \nu}\f$ measured by an observer
/// comoving with the fluid with 4-velocity \f$u_\nu\f$ where \f$F^{\mu \nu}\f$
/// is the Faraday tensor.
template <typename DataType, size_t Dim, typename Fr>
struct ComovingMagneticField : db::SimpleTag {
  using type = tnsr::A<DataType, Dim, Fr>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "ComovingMagneticField";
  }
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

/// The Lorentz factor \f$W\f$.
template <typename DataType>
struct LorentzFactor : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "LorentzFactor"; }
};

/// The magnetic field \f$B^i = n_\mu F^{i \mu}\f$ measured by an Eulerian
/// observer, where \f$n_\mu\f$ is the normal to the spatial hypersurface and
/// \f$F^{\mu \nu}\f$ is the Faraday tensor.
template <typename DataType, size_t Dim, typename Fr>
struct MagneticField : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Fr>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "MagneticField";
  }
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

/// The spatial velocity \f$v^i\f$.
template <typename DataType, size_t Dim, typename Fr>
struct SpatialVelocity : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Fr>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "SpatialVelocity";
  }
};

/// The spatial velocity one-form \f$v_i\f$.
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
}  // namespace hydro
