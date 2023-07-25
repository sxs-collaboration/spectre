// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace ForceFree {

/*!
 * \brief Identify different types of GRFFE simulations.
 *
 * In the ForceFree evolution system, we may need to execute slightly different
 * sets of evolution step actions depending on the specific type of initial
 * data. The main purpose of adding this enum variable is to make grouping
 * a specific set of initial data much easier.
 *
 */
enum SimulationType {
  //
  // Cases that do not need any special treatments. e.g. 1D wave tests, 3D black
  // hole tests.
  //
  FreeSpace,

  //
  // Simulating the magnetosphere of an isolated neutron star. In this case, we
  // need to impose the MHD condition inside the neutron star by correcting
  // evolved variables.
  //
  IsolatedNsMagnetosphere,
};

}  // namespace ForceFree
