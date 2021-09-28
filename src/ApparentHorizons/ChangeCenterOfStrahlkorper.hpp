// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

/// \cond
template <typename Frame>
class Strahlkorper;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

/// Changes the expansion center of a Strahlkorper, where the
/// expansion center is defined as the point about which the spectral
/// basis of the Strahlkorper is expanded, which is the quantity
/// returned by `Strahlkorper::center()`.
template <typename Frame>
void change_expansion_center_of_strahlkorper(
    gsl::not_null<Strahlkorper<Frame>*> strahlkorper,
    const std::array<double, 3>& new_center);

/// Changes the expansion center of a Strahlkorper to the physical
/// center.  Because `Strahlkorper::physical_center()` returns only an
/// approximate quantity,
/// `change_expansion_center_of_strahlkorper_to_physical` is
/// iterative, and does not return exactly the same result as passing
/// `Strahlkorper::physical_center()` to
/// `change_expansion_center_of_strahlkorper`.
template <typename Frame>
void change_expansion_center_of_strahlkorper_to_physical(
    gsl::not_null<Strahlkorper<Frame>*> strahlkorper);
