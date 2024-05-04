/*
 *  This file is part of libc_utils.
 *
 *  libc_utils is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  libc_utils is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with libc_utils; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/*
 *  libc_utils is being developed at the Max-Planck-Institut fuer Astrophysik
 *  and financially supported by the Deutsches Zentrum fuer Luft- und Raumfahrt
 *  (DLR).
 */

/*
 *  Functionality for measuring memory consumption
 *
 *  Copyright (C) 2012 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#include <stdio.h>
#include <string.h>
#include "memusage.h"

double residentSetSize(void)
  {
  FILE *statm = fopen("/proc/self/statm","r");
  double res;
  if (!statm) return -1.0;
  if (fscanf(statm,"%*f %lf",&res))
      { fclose(statm); return -1.0; }
  fclose(statm);
  return (res*4096);
  }

double VmHWM(void)
  {
  char word[1024];
  FILE *f = fopen("/proc/self/status", "r");
  double res;
  if (!f) return -1.0;
  while(1)
    {
    if (fscanf (f,"%1023s",word)<0)
      { fclose(f); return -1.0; }
    if (!strncmp(word, "VmHWM:", 6))
      {
      if (fscanf(f,"%lf%2s",&res,word)<0)
	{ fclose(f); return -1.0; }
      if (strncmp(word, "kB", 2))
        { fclose(f); return -1.0; }
      res *=1024;
      fclose(f);
      return res;
      }
    }
  }
