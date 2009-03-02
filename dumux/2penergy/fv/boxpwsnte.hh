// $Id$

#ifndef DUNE_BOXPWSNTE_HH
#define DUNE_BOXPWSNTE_HH

#include<dune/istl/operators.hh>
#include<dune/istl/solvers.hh>
#include<dune/istl/preconditioners.hh>

//#include <dune/common/array.hh>        // defines simple array class
#include <dune/common/fixedarray.hh>   // defines simple array classes
#include <dune/common/geometrytype.hh>
#include <dune/grid/sgrid.hh>          // a complete structured grid
#include <dune/grid/common/referenceelements.hh>
#include <dune/grid/common/gridinfo.hh>
#include <dune/grid/common/universalmapper.hh>
#include <dune/grid/common/quadraturerules.hh>
#include <dune/common/collectivecommunication.hh>
#include <dune/istl/io.hh>
#include <dune/common/timer.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/vbvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/io.hh>
#include <dune/istl/gsetc.hh>
#include <dune/istl/ilu.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/scalarproducts.hh>
#include <dune/istl/paamg/amg.hh>
#include <dune/grid/common/scsgmapper.hh>
#include <dune/grid/common/mcmgmapper.hh>
#include <dune/disc/functions/functions.hh>
#include "dumux/operators/p1operatorextended.hh"
#include <dune/disc/operators/boundaryconditions.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>
#include <dune/istl/paamg/amg.hh>
#include "dumux/pardiso/pardiso.hh"
#include "dumux/pardiso/identity.hh"
#include "dumux/nonlinear/newtonmethod.hh"
#include "dumux/2penergy/2penergymodel.hh"
#include "dumux/2penergy/2penergyproblem.hh"
#include "dumux/2penergy/fv/boxpwsntejacobian.hh"
namespace Dune
{

/**
   \brief Two phase model with Pw and Sn as primary unknowns

   This implements a two phase model with Pw and Sn as primary unknowns.
*/
template<class G, class RT>
class BoxPwSnTe
    : public LeafP1TwoPhaseModel<G, RT, TwoPhaseHeatProblem<G, RT>, BoxPwSnTeJacobian<G, RT> >
{






public:
    // define the problem type (also change the template argument above)
    typedef TwoPhaseHeatProblem<G, RT> ProblemType;

    // define the local Jacobian (also change the template argument above)
    typedef BoxPwSnTeJacobian<G, RT> LocalJacobian;

    typedef LeafP1TwoPhaseModel<G, RT, ProblemType, LocalJacobian> LeafP1TwoPhaseModel;

    typedef typename LeafP1TwoPhaseModel::FunctionType FunctionType;

    typedef typename G::Traits::LeafIndexSet IS;

    enum{m = 3};

    typedef BoxPwSnTe<G, RT> ThisType;
    typedef typename LeafP1TwoPhaseModel::FunctionType::RepresentationType VectorType;
    typedef typename LeafP1TwoPhaseModel::OperatorAssembler::RepresentationType MatrixType;
    typedef MatrixAdapter<MatrixType,VectorType,VectorType> Operator;
#ifdef HAVE_PARDISO
    SeqPardiso<MatrixType,VectorType,VectorType> pardiso;
#endif

    BoxPwSnTe(const G& g, ProblemType& prob)
        : LeafP1TwoPhaseModel(g, prob)
    {     }

    void solve()
    {




        Operator op(*(this->A));  // make operator out of matrix
        double red=1E-8;

#ifdef HAVE_PARDISO
        //    SeqPardiso<MatrixType,VectorType,VectorType> ilu0(*(this->A));
        pardiso.factorize(*(this->A));
        BiCGSTABSolver<VectorType> solver(op,pardiso,red,100,2);         // an inverse operator
        //    SeqILU0<MatrixType,VectorType,VectorType> ilu0(*(this->A),1.0);// a precondtioner
        //LoopSolver<VectorType> solver(op, ilu0, red, 10, 2);
#else
        SeqILU0<MatrixType,VectorType,VectorType> ilu0(*(this->A),1.0);// a precondtioner

        //SeqIdentity<MatrixType,VectorType,VectorType> ilu0(*(this->A));// a precondtioner
        BiCGSTABSolver<VectorType> solver(op,ilu0,red,10000,1);         // an inverse operator
#endif
        InverseOperatorResult r;
        solver.apply(*(this->u), *(this->f), r);

        return;
    }

    void update (double& dt)
    {
        this->localJacobian().setDt(dt);
        this->localJacobian().setOldSolution(this->uOldTimeStep);
        NewtonMethod<G, ThisType> newtonMethod(this->grid, *this, 1.e-8, 1.e+5);
        newtonMethod.execute();
        dt = this->localJacobian().getDt();
        double upperMass, oldUpperMass;
        double totalMass = this->injected(upperMass, oldUpperMass);
        std::cout << "total CO2 Mass: "<<totalMass << "\t" << std::endl;
        //        double MassFlux = 0;
        //        double upperValue[2], lowerValue[2];
        //        upperValue[0] = upperValue[1] = 0.;
        //        lowerValue[0] = lowerValue[1] = 0.;
        //        MassFlux = this->ComputeFlux(upperValue,lowerValue);
        //        std::cout << MassFlux <<" "<< upperValue[0] << " "<< upperValue[1] << " "<<
        //        lowerValue[0] <<" "<< lowerValue[1] <<" ";
        *(this->uOldTimeStep) = *(this->u);

        return;
    }

};

}
#endif
