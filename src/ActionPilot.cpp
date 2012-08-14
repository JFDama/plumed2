/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2012 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed-code.org for more information.

   This file is part of plumed, version 2.0.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "ActionPilot.h"

using namespace PLMD;
using namespace std;

void ActionPilot::registerKeywords(Keywords& keys){
}

ActionPilot::ActionPilot(const ActionOptions&ao):
Action(ao),
stride(1)
{
  plumed_massert( keywords.exists("STRIDE"), "when you create a new actionPilot object you must register the keyword "
                                             "STRIDE and provide a description of what the action "
                                             "is doing when it is called");
  parse("STRIDE",stride);
  log.printf("  with stride %d\n",stride);
}

bool ActionPilot::onStep()const{
  return getStep()%stride==0;
}

int ActionPilot::getStride()const{
  return stride;
}


