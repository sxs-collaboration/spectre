// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "PointwiseFunctions/SpecialRelativity/Tags.hpp"

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

/// The vector \f$J^i\f$ in \f$\dot{M} = -\int J^i s_i d^2S\f$,
/// representing the mass flux through a surface with normal \f$s_i\f$.
///
/// Note that the integral is understood
/// as a flat-space integral: all metric factors are included in \f$J^i\f$.
/// In particular, if the integral is done over a Strahlkorper, the
/// `StrahlkorperGr::euclidean_area_element` of the Strahlkorper should be used,
/// and \f$s_i\f$ is
/// the normal one-form to the Strahlkorper normalized with the flat metric,
/// \f$s_is_j\delta^{ij}=1\f$.
///
/// The formula is
/// \f$ J^i = \rho W \sqrt{\gamma}(\alpha v^i-\beta^i)\f$,
/// where \f$\rho\f$ is the mass density, \f$W\f$ is the Lorentz factor,
/// \f$v^i\f$ is the spatial velocity of the fluid,
/// \f$\gamma\f$ is the determinant of the 3-metric \f$\gamma_{ij}\f$,
/// \f$\alpha\f$ is the lapse, and \f$\beta^i\f$ is the shift.
template <typename DataType, size_t Dim, typename Fr>
struct MassFlux : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Fr>;
  static std::string name() noexcept {
    return Frame::prefix<Fr>() + "MassFlux";
  }
};
}  // namespace Tags

/// The tags for the primitive variables for GRMHD.
template <typename DataType>
using grmhd_tags =
    tmpl::list<hydro::Tags::RestMassDensity<DataType>,
               hydro::Tags::SpecificInternalEnergy<DataType>,
               sr::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>,
               hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>,
               hydro::Tags::DivergenceCleaningField<DataType>,
               sr::Tags::LorentzFactor<DataType>,
               hydro::Tags::Pressure<DataType>,
               hydro::Tags::SpecificEnthalpy<DataType>>;
}  // namespace hydro
