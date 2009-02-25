// $Id$

#ifndef DUNE_MIMETICDIFFUSION_HH
#define DUNE_MIMETICDIFFUSION_HH

#include <dune/common/helpertemplates.hh>
#include <dune/common/typetraits.hh>
#include <dune/grid/common/mcmgmapper.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>
#include <dune/grid/utility/intersectiongetter.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/preconditioners.hh>
#include "dumux/diffusion/diffusion_deprecated.hh"
#include "dumux/operators/mimeticoperator.hh"
#include "dumux/diffusion/mimetic/mimeticgroundwater.hh"
//#include "dumux/transport/problems/simpleproblem.hh"
//#include "dumux/diffusion/problems/uniformproblem.hh"

/**
 * @file
 * @brief  Base class for defining an instance of a numerical diffusion model
 * @author Bernd Flemisch
 */

namespace Dune
{
  //! \ingroup diffusion
  //! Base class for defining an instance of a numerical diffusion model.
  /*! An interface for defining a numerical diffusion model for the
   *  solution of equations of the form
   * \f$ - \text{div}\, (\lambda K \text{grad}\, p ) = 0, \f$,
   * \f$p = g\f$ on \f$\Gamma_1\f$, and \f$\lambda K \text{grad}\, p = J\f$
   * on \f$\Gamma_2\f$. Here,
   * \f$p\f$ denotes the pressure, \f$K\f$ the absolute permeability,
   * and \f$\lambda\f$ the total mobility, possibly depending on the
   * saturation.
    Template parameters are:

    - Grid      a DUNE grid type
    - RT        type used for return values
   */
  template<class G, class RT, class VC, class LocalStiffnessType = Dune::MimeticGroundwaterEquationLocalStiffness<G,RT,VC> >
  class MimeticDiffusion
  : public DeprecatedDiffusion< G, RT, VC>
  {
      template<int dim>
      struct ElementLayout
      {
          bool contains (Dune::GeometryType gt)
          {
              return gt.dim() == dim;
          }
      };

      typedef Dune::LevelCRFunction<G,RT,1> TraceType;
      typedef Dune::LevelP0Function<G,RT,2*G::dimension> NormalVelType;
      typedef Dune::MimeticOperatorAssembler<G,RT,1> LevelOperatorAssembler;

  public:
    typedef BlockVector< Dune::FieldVector<RT,1> > RepresentationType;

    void assemble(const RT t=0)
    {
        LocalStiffnessType lstiff(this->diffproblem, false, this->grid(), levell);
        A.assemble(lstiff, pressTrace, f);
        return;
    }

    void solve()
    {
        typedef typename LevelCRFunction<G,RT>::RepresentationType VectorType;
        typedef typename LevelCROperatorAssembler<G,RT,1>::RepresentationType MatrixType;
        typedef MatrixAdapter<MatrixType,VectorType,VectorType> Operator;

        //printmatrix(std::cout, *A, "global stiffness matrix", "row", 11, 3);
        //printvector(std::cout, *f, "right hand side", "row", 200, 1, 5);
        Operator op(*A);  // make operator out of matrix
        double red=1E-12;
        SeqILU0<MatrixType,VectorType,VectorType> ilu0(*A,1.0);// a precondtioner
        //SeqJac<MatrixType,VectorType,VectorType> ilu0(*A,1,0.9);// a precondtioner
        //SeqPardiso<MatrixType,VectorType,VectorType> ilu0(*A);// a precondtioner
        BiCGSTABSolver<VectorType> solver(op,ilu0,red,10000,1);         // an inverse operator
        //CGSolver<VectorType> solver(op,ilu0,red,10000,1);         // an inverse operator
        InverseOperatorResult r;
        solver.apply(*pressTrace, *f, r);
        //printvector(std::cout, *pressTrace, "solution", "row", 200, 1, 5);
        return;
    }

    void postprocess()
    {
        LocalStiffnessType lstiff(this->diffproblem, false, this->grid(), levell);
        A.calculatePressure(lstiff, pressTrace, normalVelocity, this->diffproblem.variables.pressure);
        //printvector(std::cout, this->variables.pressure, "element pressures", "row", 200, 1, 5);
        //printvector(std::cout, *normalVelocity, "normal velocities", "row", 200, 1, 5);
        return;
    }

    void pressure(const RT t=0)
    {
        assemble(t);
        solve();
        postprocess();
        return;
    }

    void calcTotalVelocity(const RT t=0) const
    {
        // ASSUMES axiparallel grids in 2D
        for (int i = 0; i < this->grid().size(levell, 0); i++) {
            this->diffproblem.variables.velocity[i][0][0] = -(*normalVelocity)[i][0];
            this->diffproblem.variables.velocity[i][0][1] = 0;
            this->diffproblem.variables.velocity[i][1][0] = (*normalVelocity)[i][1];
            this->diffproblem.variables.velocity[i][1][1] = 0;
            this->diffproblem.variables.velocity[i][2][0] = 0;
            this->diffproblem.variables.velocity[i][2][1] = -(*normalVelocity)[i][2];
            this->diffproblem.variables.velocity[i][3][0] = 0;
            this->diffproblem.variables.velocity[i][3][1] = (*normalVelocity)[i][3];
        }
        return;
    }

    void calcTotalVelocity(const RT t, double lev) const
    {
        DUNE_THROW(Dune::NotImplemented, "upscaled velocities only implemented in FVDiffusion");
    }

    void vtkout (const char* name, int k) const
    {
        VTKWriter<typename G::LevelGridView>
            vtkwriter(this->grid().levelView(this->level()));
        char fname[128];
        sprintf(fname,"%s-%05d",name,k);
        vtkwriter.addCellData(this->diffproblem.variables.pressure,"total pressure p~");
        vtkwriter.write(fname, VTKOptions::ascii);
    }

    MimeticDiffusion(G& g, DeprecatedDiffusionProblem<G, RT,VC>& prob, int lev = 0, bool calcPressure = true)
      : DeprecatedDiffusion<G, RT, VC>(g, prob, lev), levell(lev),
      pressTrace(g, levell), normalVelocity(g, levell), f(g, levell), A(g, levell)
    {
        *pressTrace = 0;
        *f = 0;
    }

//  private:
      int levell;
      TraceType pressTrace;         //!< vector of pressure traces
      NormalVelType normalVelocity;
      TraceType f;
      LevelOperatorAssembler A;
  };

}
#endif
