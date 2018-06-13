// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

class DataVector;

/// \ingroup EvolutionSystemsGroup
/// \brief Items related to general relativistic magnetohydrodynamics (GRMHD)
namespace grmhd {
/// %Tags for general relativistic magnetohydrodynamics.
namespace Tags {

/// The rest-mass density \f$\rho\f$
struct RestMassDensity : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "RestMassDensity"; }
};

/// The specific internal energy \f$\epsilon\f$
struct SpecificInternalEnergy : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "SpecificInternalEnergy"; }
};

/// The spatial velocity \f$v^i\f$
template <typename Fr = Frame::Inertial>
struct SpatialVelocity : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Fr>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "SpatialVelocity";
  }
};

/// The magnetic field \f$B^i = n_\mu F^{i \mu}\f$ measured by an Eulerian
/// observer where \f$n_\mu\f$ is the normal to the spatial hypersurface and
/// \f$F^{\mu \nu}\f$ is the Faraday tensor
template <typename Fr = Frame::Inertial>
struct MagneticField : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Fr>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "MagneticField";
  }
};

/// The divergence-cleaning field \f$\Phi\f$
struct DivergenceCleaningField : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "DivergenceCleaningField"; }
};

/// The fluid pressure \f$p\f$
struct Pressure : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Pressure"; }
};

/// The specific enthalpy  \f$h\f$
struct SpecificEnthalpy : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "SpecificEnthalpy"; }
};

/// The magnetic field \f$b^\mu = u_\nu F^{\mu \nu}\f$ measured by an observer
/// comoving with the fluid with 4-velocity \f$u_\nu\f$ where \f$F^{\mu \nu}\f$
/// is the Faraday tensor
template <typename Fr = Frame::Inertial>
struct ComovingMagneticField : db::SimpleTag {
  using type = tnsr::A<DataVector, 3, Fr>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "ComovingMagneticField";
  }
};

/// The magnetic pressure \f$p_m\f$
struct MagneticPressure : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "MagneticPressure"; }
};

/// The Lorentz factor \f$W\f$
struct LorentzFactor : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "LorentzFactor"; }
};

}  // namespace Tags
}  // namespace grmhd
