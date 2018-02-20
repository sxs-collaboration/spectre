# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_DOMAIN_CREATORS
  Domain/DomainCreators/Test_Brick.cpp
  Domain/DomainCreators/Test_Disk.cpp
  Domain/DomainCreators/Test_Interval.cpp
  Domain/DomainCreators/Test_Rectangle.cpp
  Domain/DomainCreators/Test_RotatedIntervals.cpp
  Domain/DomainCreators/Test_Shell.cpp
  Domain/DomainCreators/Test_Sphere.cpp
  )

set(SPECTRE_UNIT_DOMAIN
  "${SPECTRE_UNIT_DOMAIN};${SPECTRE_UNIT_DOMAIN_CREATORS}"
  PARENT_SCOPE)
set(LIBRARIES_TO_LINK "${LIBRARIES_TO_LINK};DomainCreators" PARENT_SCOPE)
