// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function mass.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
template <size_t>
class Index;
/// \endcond

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Applies the mass matrix.
 *
 * \details We use a diagonal mass matrix approximation. Please refer to
 * dg::lift_flux() for details.
 */
template <size_t Dim>
DataVector mass(
    const DataVector& data,
    const Jacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>& jacobian,
    const Index<Dim>& mesh) noexcept;
