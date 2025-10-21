// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace gr {

/*!
 * \brief Computes the tortoise coordinates radius from the Boyer-Lindquist
 * radius.
 *
 * This function evaluates the transformation from tortoise coordinates $r_*$ to
 * Boyer-Lindquist radius $r$:
 * \begin{equation}
 * r_* = r + \frac{2 M}{r_+ - r_-}\left(
 *   r_+ \ln(\frac{r - r_+}{2 M}) - r_- \ln(\frac{r - r_-}{2 M}) \right)
 * \end{equation}
 * where $r_\pm = M \pm \sqrt{M^2 - a^2}$.
 *
 * This transformation has the Jacobian
 * \begin{equation}
 *   \frac{dr_*}{dr} = \frac{r^2 + a^2}{\Delta}
 * \end{equation}
 * with $\Delta = r^2 - 2 M r + a^2$.
 *
 * \param r_minus_r_plus Boyer-Lindquist radius minus $r_+$: $r - r_+$.
 * \param mass Kerr mass parameter $M$.
 * \param dimensionless_spin Kerr dimensionless spin parameter $\chi=a/M$.
 * \return Tortoise coordinates radius $r_*$.
 */
template <typename DataType>
DataType tortoise_radius_from_boyer_lindquist_minus_r_plus(
    const DataType& r_minus_r_plus, double mass, double dimensionless_spin);

/*!
 * \brief Computes the Boyer-Lindquist radius from tortoise coordinates.
 *
 * This function inverts the transformation from tortoise coordinates radius
 * $r_*$ to Boyer-Lindquist radius $r$:
 * \begin{equation}
 * r_* = r + \frac{2 M}{r_+ - r_-}\left(
 *   r_+ \ln(\frac{r - r_+}{2 M}) - r_- \ln(\frac{r - r_-}{2 M}) \right)
 * \end{equation}
 * where $r_\pm = M \pm \sqrt{M^2 - a^2}$.
 *
 * It performs a numerical rootfind to invert the above equation.
 *
 * \param r_star Tortoise coordinate $r_*$.
 * \param mass Kerr mass parameter $M$.
 * \param dimensionless_spin Kerr dimensionless spin parameter $\chi=a/M$.
 * \return Boyer-Lindquist radius minus $r_+$: $r - r_+$.
 */
template <typename DataType>
DataType boyer_lindquist_radius_minus_r_plus_from_tortoise(
    const DataType& r_star, double mass, double dimensionless_spin);

}  // namespace gr
