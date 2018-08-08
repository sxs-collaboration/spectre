# Distributed under the MIT License.
# See LICENSE.txt for details.

set(MY_LIBRARY_SOURCES
  StepControllers/BinaryFraction.cpp
  StepControllers/FullSlab.cpp
  StepControllers/SimpleTimes.cpp
  StepControllers/SplitRemaining.cpp
  )

set(LIBRARY_SOURCES "${LIBRARY_SOURCES};${MY_LIBRARY_SOURCES}" PARENT_SCOPE)
