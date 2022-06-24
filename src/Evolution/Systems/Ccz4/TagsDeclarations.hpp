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
template <typename DataType = DataVector>
struct ConformalFactor;
template <typename DataType = DataVector>
struct ConformalFactorSquared;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct ATilde;
template <typename DataType = DataVector>
struct TraceATilde;
template <typename DataType = DataVector>
struct LogLapse;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct FieldA;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct FieldB;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct FieldD;
template <typename DataType = DataVector>
struct LogConformalFactor;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct FieldP;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct FieldDUp;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct ConformalChristoffelSecondKind;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct DerivConformalChristoffelSecondKind;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct ChristoffelSecondKind;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct Ricci;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct GradGradLapse;
template <typename DataType = DataVector>
struct DivergenceLapse;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct ContractedConformalChristoffelSecondKind;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct DerivContractedConformalChristoffelSecondKind;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct GammaHat;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct SpatialZ4Constraint;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct SpatialZ4ConstraintUp;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct GradSpatialZ4Constraint;
template <typename DataType = DataVector>
struct RicciScalarPlusDivergenceZ4Constraint;
// Temporary expressions for computing above quantities of interest
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct GammaHatMinusContractedConformalChristoffel;
template <typename DataType = DataVector>
struct KMinus2ThetaC;
template <typename DataType = DataVector>
struct KMinusK0Minus2ThetaC;
template <typename DataType = DataVector>
struct ContractedFieldB;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct ConformalMetricTimesFieldB;
template <typename DataType = DataVector>
struct LapseTimesRicciScalarPlus2DivergenceZ4Constraint;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct ConformalMetricTimesTraceATilde;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct LapseTimesATilde;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct FieldDUpTimesATilde;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct LapseTimesDerivATilde;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct InverseConformalMetricTimesDerivATilde;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct ATildeMinusOneThirdConformalMetricTimesTraceATilde;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct LapseTimesFieldA;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct ShiftTimesDerivGammaHat;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct InverseTauTimesConformalMetric;
template <typename DataType = DataVector>
struct LapseTimesSlicingCondition;
}  // namespace Tags

/// \brief Input option tags for the CCZ4 evolution system
namespace OptionTags {
struct Group;
}  // namespace OptionTags
}  // namespace Ccz4
