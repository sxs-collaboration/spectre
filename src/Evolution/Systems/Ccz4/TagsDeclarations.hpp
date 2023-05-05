// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"

class DataVector;

namespace Ccz4 {
/// \brief Tags for the CCZ4 formulation of Einstein equations
namespace Tags {
// Quantities of interest
template <typename DataType>
struct ConformalFactor;
template <typename DataType>
struct ConformalFactorSquared;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct ATilde;
template <typename DataType>
struct TraceATilde;
template <typename DataType>
struct LogLapse;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct FieldA;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct FieldB;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct FieldD;
template <typename DataType>
struct LogConformalFactor;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct FieldP;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct FieldDUp;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct ConformalChristoffelSecondKind;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct DerivConformalChristoffelSecondKind;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct ChristoffelSecondKind;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct Ricci;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct GradGradLapse;
template <typename DataType>
struct DivergenceLapse;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct ContractedConformalChristoffelSecondKind;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct DerivContractedConformalChristoffelSecondKind;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct GammaHat;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct SpatialZ4Constraint;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct SpatialZ4ConstraintUp;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct GradSpatialZ4Constraint;
template <typename DataType>
struct RicciScalarPlusDivergenceZ4Constraint;
// Temporary expressions for computing above quantities of interest
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct GammaHatMinusContractedConformalChristoffel;
template <typename DataType>
struct KMinus2ThetaC;
template <typename DataType>
struct KMinusK0Minus2ThetaC;
template <typename DataType>
struct ContractedFieldB;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct ConformalMetricTimesFieldB;
template <typename DataType>
struct LapseTimesRicciScalarPlus2DivergenceZ4Constraint;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct ConformalMetricTimesTraceATilde;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct LapseTimesATilde;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct FieldDUpTimesATilde;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct LapseTimesDerivATilde;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct InverseConformalMetricTimesDerivATilde;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct ATildeMinusOneThirdConformalMetricTimesTraceATilde;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct LapseTimesFieldA;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct ShiftTimesDerivGammaHat;
template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
struct InverseTauTimesConformalMetric;
template <typename DataType>
struct LapseTimesSlicingCondition;
}  // namespace Tags

/// \brief Input option tags for the CCZ4 evolution system
namespace OptionTags {
struct Group;
}  // namespace OptionTags
}  // namespace Ccz4
