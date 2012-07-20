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

#include "FunctionVessel.h"
#include "ActionWithDistribution.h"

namespace PLMD {

class sum : public SumVessel {
public:
  static void reserveKeyword( Keywords& keys );
  sum( const VesselOptions& da );
  double compute( const unsigned& i, const double& val, double& df );
};

PLUMED_REGISTER_VESSEL(sum,"SUM")

void sum::reserveKeyword( Keywords& keys ){
  keys.reserveFlag("SUM",false,"calculate the sum of all the quantities and store it in a value called keyword.sum.");
}

sum::sum( const VesselOptions& da ) :
SumVessel(da)
{
  addOutput("sum");
  log.printf("  value %s.sum contains the sum of all the values\n",(getAction()->getLabel()).c_str());
}

double sum::compute( const unsigned& i, const double& val, double& df ){
  plumed_assert( i==0 );
  df=1.0; return val;
}

}
