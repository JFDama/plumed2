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
#include "SwitchingFunction.h"
#include "ActionWithDistribution.h"

namespace PLMD {

class less_than : public SumVessel {
private:
  SwitchingFunction sf;
public:
  static void reserveKeyword( Keywords& keys ); 
  less_than( const VesselOptions& da );
  double compute( const unsigned& i, const double& val, double& df ); 
  void printKeywords( Log& log );
};

PLUMED_REGISTER_VESSEL(less_than,"LESS_THAN")

void less_than::reserveKeyword( Keywords& keys ){
  keys.reserve("optional","LESS_THAN", "take the number of variables less than the specified target and "
                                       "store it in a value called lt<target>. " + SwitchingFunction::documentation() );
}

less_than::less_than( const VesselOptions& da ) :
SumVessel(da)
{
  std::string errormsg; sf.set( da.parameters, errormsg ); 
  if( errormsg.size()!=0 ) error( errormsg ); 
  std::string vv; Tools::convert( sf.get_r0(), vv );
  addOutput("lt" + vv);
  log.printf("  value %s.lt%s contains number of values less than %s\n",(getAction()->getLabel()).c_str(),vv.c_str(),(sf.description()).c_str() );
}

void less_than::printKeywords( Log& log ){
  sf.printKeywords( log );
}

double less_than::compute( const unsigned& i, const double& val, double& df ){
  plumed_assert( i==0 );
  double f; f = sf.calculate(val, df); df*=val;
  return f;
}

}
