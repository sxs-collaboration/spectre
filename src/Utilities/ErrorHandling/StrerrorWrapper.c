/*
 * Distributed under the MIT License.
 * See LICENSE.txt for details.
 */

#include "Utilities/ErrorHandling/StrerrorWrapper.h"

#include <stddef.h>
#include <string.h>

int spectre_strerror_r(const int errnum, char* const buf, const size_t buflen) {
  return strerror_r(errnum, buf, buflen);
}
