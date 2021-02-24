// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace evolution::dg {
/*!
 * \ingroup DgSubcellGroup
 * \brief Implementation of a generic finite volume/conservative finite
 * difference subcell limiter
 *
 * Our implementation of a finite volume (FV) or finite difference (FD) subcell
 * limiter (SCL) follows \cite Dumbser2014a. Other implementations of a subcell
 * limiter exist, e.g. \cite Sonntag2014 \cite Casoni2012 \cite Hou2007. Our
 * implementation and that of \cite Dumbser2014a are a generalization of the
 * Multidimensional Optimal Order Detection (MOOD) algorithm \cite CLAIN20114028
 * \cite DIOT201243 \cite Diot2013 \cite Loubere2014.
 */
namespace subcell {
/*!
 * \ingroup DgSubcellGroup
 * \brief Code specific to a finite volume subcell limiter
 */
namespace fv {}
/*!
 * \ingroup DgSubcellGroup
 * \brief Code specific to a conservative finite difference subcell limiter
 */
namespace fd {}
}  // namespace subcell
}  // namespace evolution::dg
