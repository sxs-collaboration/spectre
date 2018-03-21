// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions useful for testing curved scalar wave

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"

template <typename DataType>
Scalar<DataType> make_pi(const DataType& used_for_size);

template <size_t Dim, typename DataType>
tnsr::i<DataType, Dim> make_phi(const DataType& used_for_size);

template <size_t Dim, typename DataType>
tnsr::i<DataType, Dim> make_d_psi(const DataType& used_for_size);

template <size_t Dim, typename DataType>
tnsr::i<DataType, Dim> make_d_pi(const DataType& used_for_size);

template <size_t Dim, typename DataType>
tnsr::ij<DataType, Dim> make_d_phi(const DataType& used_for_size);

template <typename DataType>
Scalar<DataType> make_constraint_gamma1(const DataType& used_for_size);

template <typename DataType>
Scalar<DataType> make_constraint_gamma2(const DataType& used_for_size);
