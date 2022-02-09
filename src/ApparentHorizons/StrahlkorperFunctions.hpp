// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;

template <typename Fr>
class Strahlkorper;

template <typename X, typename Symm, typename IndexList>
class Tensor;

namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

/// \ingroup SurfacesGroup
/// Contains functions that depend on a Strahlkorper but not on a metric.
namespace StrahlkorperFunctions {
/// @{
/*!
 * (Euclidean) distance \f$r_{\rm surf}(\theta,\phi)\f$ from the
 * expansion center to each point of the Strahlkorper surface.
 */
template <typename Fr>
Scalar<DataVector> radius(const Strahlkorper<Fr>& strahlkorper);

/*!
 * (Euclidean) distance \f$r_{\rm surf}(\theta,\phi)\f$ from the
 * expansion center to each point of the Strahlkorper surface.
 */
template <typename Fr>
void radius(const gsl::not_null<Scalar<DataVector>*> result,
            const Strahlkorper<Fr>& strahlkorper);
/// @}
}  // namespace StrahlkorperFunctions
