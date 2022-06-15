// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <utility>

#include "Utilities/ForceInline.hpp"

/*!
 * \ingroup UtilitiesGroup
 * \brief Used for overloading lambdas, useful for lambda-SFINAE
 *
 * \snippet Utilities/Test_Overloader.cpp overloader_example
 */
template <class... Fs>
struct Overloader : Fs... {
  using Fs::operator()...;
};

template <class... Fs>
Overloader(Fs...) -> Overloader<Fs...>;
