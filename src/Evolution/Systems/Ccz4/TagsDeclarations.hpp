// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/IndexType.hpp"

class DataVector;

namespace Ccz4 {
/// \brief Tags for the CCZ4 formulation of Einstein equations
namespace Tags {
template <typename DataType = DataVector>
struct ConformalFactor;
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
struct ConformalChristoffelSecondKind;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct ChristoffelSecondKind;
template <size_t Dim, typename Frame = Frame::Inertial,
          typename DataType = DataVector>
struct GradGradLapse;
}  // namespace Tags

/// \brief Input option tags for the generalized harmonic evolution system
namespace OptionTags {
struct Group;
}  // namespace OptionTags
}  // namespace Ccz4
