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

#include "VesselValueAccess.h"
#include "ActionWithDistribution.h"

namespace PLMD {

class moment : public VesselStoreAllValues {
private:
  Value myvalue, myvalue2;
  std::vector<unsigned> powers;
  std::vector<Value*> value_out;
public:
  static void reserveKeyword( Keywords& keys );
  moment( const VesselOptions& da );
  void finish( const double& tolerance );
  void local_resizing();
  bool applyForce( std::vector<double>& forces );
};

PLUMED_REGISTER_VESSEL(moment,"MOMENTS")

void moment::reserveKeyword( Keywords& keys ){
  std::ostringstream ostr;
  keys.reserve("optional","MOMENTS","calculate the moments of the distribution of collective variables. " 
  "The \\f$m\\f$th moment of a distribution is calculated using \\f$\\frac{1}{N} \\sum_{i=1}^N ( s_i - \\overline{s} )^m \\f$, where \\f$\\overline{s}\\f$ is "
  "the average for the distribution.  The moment keyword takes a lists of integers as input or a range.  Each integer is a value of \\f$m\\f$.");  
}

moment::moment( const VesselOptions& da) :
VesselStoreAllValues(da)
{
   ActionWithValue* a=dynamic_cast<ActionWithValue*>( getAction() );
   plumed_massert(a,"cannot create passable values as base action does not inherit from ActionWithValue");

   std::vector<std::string> moments=Tools::getWords(da.parameters,"\t\n ,"); 
   Tools::interpretRanges(moments); unsigned nn;
   for(unsigned i=0;i<moments.size();++i){
       a->addComponentWithDerivatives( "moment_" + moments[i] );  
       a->componentIsNotPeriodic( "moment_" + moments[i] );
       value_out.push_back( a->copyOutput( a->getNumberOfComponents()-1 ) );
       Tools::convert( moments[i], nn );
       if( nn<2 ) error("moments are only possible for m>=2" );
       powers.push_back( nn ); std::string num; Tools::convert(powers[i],num);
       log.printf("  value %s.moment_%s contains the %d th moment of the distribution\n",(getAction()->getLabel()).c_str(),moments[i].c_str(),powers[i]);
   }

   if( getAction()->isPeriodic() ){
      double min, max;
      getAction()->retrieveDomain( min, max );
      myvalue.setDomain( min, max );
      myvalue2.setDomain( min, max );
   } else {
       myvalue.setNotPeriodic();
       myvalue2.setNotPeriodic();
   }
}

void moment::local_resizing(){
   unsigned nder=getAction()->getNumberOfDerivatives();
   for(unsigned i=0;i<value_out.size();++i) value_out[i]->resizeDerivatives( nder );
}

void moment::finish( const double& tolerance ){
  const double pi=3.141592653589793238462643383279502884197169399375105820974944592307;
  unsigned nvals=getAction()->getNumberOfFunctionsInAction();

  double mean=0;
  if( getAction()->isPeriodic() ){
     double min, max, pfactor; getAction()->retrieveDomain( min, max );
     pfactor = 2*pi / ( max-min );
     double sinsum=0, cossum=0, val;
     for(unsigned i=0;i<nvals;++i){ val=pfactor*( getValue(i) - min ); sinsum+=sin(val); cossum+=cos(val); }
     mean = 0.5 + atan2( sinsum / static_cast<double>( nvals ) , cossum / static_cast<double>( nvals ) ) / (2*pi);
     mean = min + (max-min)*mean;
  } else {
     for(unsigned i=0;i<nvals;++i) mean+=getValue(i);    
     mean/=static_cast<double>( nvals );
  }

  for(unsigned npow=0;npow<powers.size();++npow){
     double dev1=0; 
     for(unsigned i=0;i<nvals;++i) dev1+=pow( myvalue.difference( mean, getValue(i) ), powers[npow] - 1 ); 
     dev1/=static_cast<double>( nvals );

     double pref, tmp, moment=0; 
     for(unsigned i=0;i<nvals;++i){
         getValue( i, myvalue );
         tmp=myvalue.difference( mean, myvalue.get() );
         pref=pow( tmp, powers[npow] - 1 ) - dev1;
         moment+=pow( tmp, powers[npow] );
         getAction()->mergeDerivatives( i, myvalue, pref, myvalue2 );
         add( myvalue2, value_out[npow] );
     }
     value_out[npow]->chainRule( powers[npow] / static_cast<double>( nvals ) );
     value_out[npow]->set( moment / static_cast<double>( nvals ) ); 
  }
}

bool moment::applyForce( std::vector<double>& forces ){
  std::vector<double> tmpforce( forces.size() );
  forces.assign(forces.size(),0.0); bool wasforced=false;
  for(unsigned i=0;i<value_out.size();++i){
     if( value_out[i]->applyForce( tmpforce ) ){
         wasforced=true;
         for(unsigned j=0;j<forces.size();++j) forces[j]+=tmpforce[j];
     }
  }
  return wasforced;
}

}
