// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame

class DataVector;

namespace hydro {
namespace Tags {

template <typename DataType>
struct AlfvenSpeedSquared;
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct ComovingMagneticField;
template <typename DataType>
struct ComovingMagneticFieldSquared;
template <typename DataType>
struct DivergenceCleaningField;
struct EquationOfStateBase;
template <typename EquationOfStateType>
struct EquationOfState;
template <typename DataType>
struct LorentzFactor;
template <typename DataType>
struct LorentzFactorSquared;
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct MagneticField;
template <typename DataType>
struct MagneticFieldDotSpatialVelocity;
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct MagneticFieldOneForm;
template <typename DataType>
struct MagneticFieldSquared;
template <typename DataType>
struct MagneticPressure;
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct MassFlux;
template <typename DataType>
struct Pressure;
template <typename DataType>
struct RestMassDensity;
template <typename DataType>
struct SoundSpeedSquared;
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct SpatialVelocity;
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct SpatialVelocityOneForm;
template <typename DataType>
struct SpatialVelocitySquared;
template <typename DataType>
struct SpecificEnthalpy;
template <typename DataType>
struct SpecificInternalEnergy;
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct LowerSpatialFourVelocity;
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct LorentzFactorTimesSpatialVelocity;
}  // namespace Tags
/// \endcond

/// The tags for the primitive variables for GRMHD.
template <typename DataType>
using grmhd_tags =
    tmpl::list<hydro::Tags::RestMassDensity<DataType>,
               hydro::Tags::SpecificInternalEnergy<DataType>,
               hydro::Tags::SpatialVelocity<DataType, 3>,
               hydro::Tags::MagneticField<DataType, 3>,
               hydro::Tags::DivergenceCleaningField<DataType>,
               hydro::Tags::LorentzFactor<DataType>,
               hydro::Tags::Pressure<DataType>,
               hydro::Tags::SpecificEnthalpy<DataType>>;
}  // namespace hydro
