#include "Function.h"
#include "Colvar.h"

using namespace PLMD;
using namespace std;

void Function::registerKeywords(Keywords& keys){
  Action::registerKeywords(keys);
  ActionWithValue::registerKeywords(keys);
  ActionWithArguments::registerKeywords(keys);
  keys.reserve("compulsory","PERIODIC","if the output of your function is periodic then you should specify the periodicity of the function.  If the output is not periodic you must state this using PERIODIC=NO");
}

Function::Function(const ActionOptions&ao):
Action(ao),
ActionWithValue(ao),
ActionWithArguments(ao)
{
}

void Function::addValueWithDerivatives(){
  plumed_massert( getNumberOfArguments()!=0, "for functions you must requestArguments before adding values");
  ActionWithValue::addValueWithDerivatives();  
  getPntrToValue()->resizeDerivatives(getNumberOfArguments());

  if( keywords.exists("PERIODIC") ){
     double min(0),max(0); std::vector<std::string> period;  
     parseVector("PERIODIC",period);  
     if(period.size()==1 && period[0]=="NO"){
        setNotPeriodic();
     } else if(period.size()==2 && Tools::convert(period[0],min) && Tools::convert(period[1],max)){
        setPeriodic(min,max);
     } else error("missing PERIODIC keyword");
  }
} 
  
void Function::addComponentWithDerivatives( const std::string& name ){
  plumed_massert( getNumberOfArguments()!=0, "for functions you must requestArguments before adding values");
  ActionWithValue::addComponentWithDerivatives(name);
  getPntrToComponent(name)->resizeDerivatives(getNumberOfArguments());
}

void Function::apply(){

  vector<double>   f(getNumberOfArguments(),0.0);
  bool at_least_one_forced=false;

  std::vector<double> forces( getNumberOfArguments() );
  for(int i=0;i<getNumberOfComponents();++i){
    if( getPntrToComponent(i)->applyForce( forces ) ){
       at_least_one_forced=true;
       for(unsigned j=0;j<forces.size();j++){ f[j]+=forces[j]; }
    }
  }

  if(at_least_one_forced) for(unsigned i=0;i<getNumberOfArguments();++i){
    getPntrToArgument(i)->addForce(f[i]);
  }
}
