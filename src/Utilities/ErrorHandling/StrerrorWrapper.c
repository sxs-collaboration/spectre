/*
 * Distributed under the MIT License.
 * See LICENSE.txt for details.
 */

#ifdef __linux__

// Force selection of XSI-compliant strerror_r by glibc
// Taken from https://groups.google.com/g/golang-checkins/c/cd1OSJoE20c?pli=1
#undef _POSIX_C_SOURCE
// POSIX_C_SOURCE must be >= 200112L and _GNU_SOURCE not defined
// as per strerror_r man page (https://linux.die.net/man/3/strerror_r)
#define _POSIX_C_SOURCE 200112L  // NOLINT
#undef _GNU_SOURCE

#endif /* __linux__ */

#include "Utilities/ErrorHandling/StrerrorWrapper.h"

#include <stddef.h>
#include <string.h>

int spectre_strerror_r(const int errnum, char* const buf, const size_t buflen) {
  return strerror_r(errnum, buf, buflen);
}
