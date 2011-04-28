// $Id$
/****************************************************************************
 *   Copyright (C) 2008-2011 by Andreas Lauser                               *
 *   Copyright (C) 2008-2010 by Bernd Flemisch                               *
 *   Institute of Hydraulic Engineering                                      *
 *   University of Stuttgart, Germany                                        *
 *   email: <givenname>.<name>@iws.uni-stuttgart.de                          *
 *                                                                           *
 *   This program is free software: you can redistribute it and/or modify    *
 *   it under the terms of the GNU General Public License as published by    *
 *   the Free Software Foundation, either version 2 of the License, or       *
 *   (at your option) any later version.                                     *
 *                                                                           *
 *   This program is distributed in the hope that it will be useful,         *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *   GNU General Public License for more details.                            *
 *                                                                           *
 *   You should have received a copy of the GNU General Public License       *
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.   *
 *****************************************************************************/
/*!
 * \file
 * \brief Reference implementation of a controller class for the Newton solver.
 *
 * Usually this controller should be sufficient.
 */
#ifndef DUMUX_NEWTON_CONTROLLER_HH
#define DUMUX_NEWTON_CONTROLLER_HH

#include <dumux/io/vtkmultiwriter.hh>
#include <dumux/common/exceptions.hh>
#include <dumux/common/math.hh>

#include <dumux/common/pardiso.hh>

#include <dumux/io/vtkmultiwriter.hh>

#if HAVE_DUNE_PDELAB

#include <dumux/common/pdelabpreconditioner.hh>

#else // ! HAVE_DUNE_PDELAB
#include <dune/istl/overlappingschwarz.hh>
#include <dune/istl/schwarz.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <dune/istl/io.hh>

#endif // HAVE_DUNE_PDELAB

#include <dumux/linear/vertexborderlistfromgrid.hh>
#include <dumux/linear/foreignoverlapfrombcrsmatrix.hh>
#include <dumux/linear/domesticoverlapfrombcrsmatrix.hh>
#include <dumux/linear/globalindices.hh>

#include <dumux/linear/overlappingbcrsmatrix.hh>
#include <dumux/linear/overlappingblockvector.hh>

namespace Dumux
{
namespace Properties
{
//! Specifies the implementation of the Newton controller
NEW_PROP_TAG(NewtonController);

//! Specifies the type of the actual Newton method
NEW_PROP_TAG(NewtonMethod);

//! Specifies the type of a solution
NEW_PROP_TAG(SolutionVector);

//! Specifies the type of a vector of primary variables at a degree of freedom
NEW_PROP_TAG(PrimaryVariables);

//! Specifies the type of a global Jacobian matrix
NEW_PROP_TAG(JacobianMatrix);

//! Specifies the type of the Vertex mapper
NEW_PROP_TAG(VertexMapper);

//! specifies the type of the time manager
NEW_PROP_TAG(TimeManager);

/*!
 * \brief Specifies the verbosity of the linear solver
 *
 * By default it is 0, i.e. it doesn't print anything. Setting this
 * property to 1 prints aggregated convergence rates, 2 prints the
 * convergence rate of every iteration of the scheme.
 */
NEW_PROP_TAG(NewtonLinearSolverVerbosity);

//! specifies whether the convergence rate and the global residual
//! gets written out to disk for every newton iteration (default is false)
NEW_PROP_TAG(NewtonWriteConvergence);

//! Specifies whether time step size should be increased during the
//! Newton methods first few iterations
NEW_PROP_TAG(EnableTimeStepRampUp);

//! Specifies whether the Jacobian matrix should only be reassembled
//! if the current solution deviates too much from the evaluation point
NEW_PROP_TAG(EnablePartialReassemble);

/*!
 * \brief Specifies whether the update should be done using the line search
 *        method instead of the plain Newton method.
 *
 * Whether this property has any effect depends on wether the line
 * search method is implemented for the actual model's Newton
 * controller's update() method. By default line search is not used.
 */
NEW_PROP_TAG(NewtonUseLineSearch);

SET_PROP_DEFAULT(NewtonLinearSolverVerbosity)
{public:
    static const int value = 0;
};

SET_PROP_DEFAULT(NewtonWriteConvergence)
{public:
    static const bool value = false;
};

SET_PROP_DEFAULT(NewtonUseLineSearch)
{public:
    static const bool value = false;
};
};

//! \cond INTERNAL
/*!
 * \brief Writes the intermediate solutions during
 *        the Newton scheme
 */
template <class TypeTag, bool enable>
struct NewtonConvergenceWriter
{
    typedef typename GET_PROP_TYPE(TypeTag, PTAG(GridView)) GridView;
    typedef typename GET_PROP_TYPE(TypeTag, PTAG(NewtonController)) NewtonController;

    typedef typename GET_PROP_TYPE(TypeTag, PTAG(SolutionVector)) SolutionVector;
    typedef typename GET_PROP_TYPE(TypeTag, PTAG(JacobianMatrix)) JacobianMatrix;

    typedef Dumux::VtkMultiWriter<GridView>  VtkMultiWriter;

    NewtonConvergenceWriter(NewtonController &ctl)
        : ctl_(ctl)
    {
        timeStepIndex_ = 0;
        iteration_ = 0;
        vtkMultiWriter_ = new VtkMultiWriter("convergence");
    }

    ~NewtonConvergenceWriter()
    { delete vtkMultiWriter_; };

    void beginTimestep()
    {
        ++timeStepIndex_;
        iteration_ = 0;
    };

    void beginIteration(const GridView &gv)
    {
        ++ iteration_;
        vtkMultiWriter_->beginTimestep(timeStepIndex_ + iteration_ / 100.0,
                                       gv);
    };

    void writeFields(const SolutionVector &uLastIter,
                     const SolutionVector &deltaU)
    {
        ctl_.method().model().addConvergenceVtkFields(*vtkMultiWriter_, uLastIter, deltaU);
    };

    void endIteration()
    { vtkMultiWriter_->endTimestep(); };

    void endTimestep()
    {
        ++timeStepIndex_;
        iteration_ = 0;
    };

private:
    int timeStepIndex_;
    int iteration_;
    VtkMultiWriter *vtkMultiWriter_;
    NewtonController &ctl_;
};

/*!
 * \brief Writes the intermediate solutions during
 *        the Newton scheme.
 *
 * This is the dummy specialization for the case where we don't want
 * to do anything.
 */
template <class TypeTag>
struct NewtonConvergenceWriter<TypeTag, false>
{
    typedef typename GET_PROP_TYPE(TypeTag, PTAG(GridView)) GridView;
    typedef typename GET_PROP_TYPE(TypeTag, PTAG(NewtonController)) NewtonController;
    typedef typename GET_PROP_TYPE(TypeTag, PTAG(SolutionVector)) SolutionVector;
   
    typedef Dumux::VtkMultiWriter<GridView>  VtkMultiWriter;

    NewtonConvergenceWriter(NewtonController &ctl)
    {};

    void beginTimestep()
    { };

    void beginIteration(const GridView &gv)
    { };

    void writeFields(const SolutionVector &uLastIter,
                     const SolutionVector &deltaU)
    { };

    void endIteration()
    { };

    void endTimestep()
    { };
};
//! \endcond

/*!
 * \brief A reference implementation of a newton controller specific
 *        for the box scheme.
 *
 * If you want to specialize only some methods but are happy with the
 * defaults of the reference controller, derive your controller from
 * this class and simply overload the required methods.
 */
template <class TypeTag>
class NewtonController
{
    typedef typename GET_PROP_TYPE(TypeTag, PTAG(Scalar)) Scalar;
    typedef typename GET_PROP_TYPE(TypeTag, PTAG(NewtonController)) Implementation;
    typedef typename GET_PROP_TYPE(TypeTag, PTAG(GridView)) GridView;

    typedef typename GET_PROP_TYPE(TypeTag, PTAG(Problem)) Problem;
    typedef typename GET_PROP_TYPE(TypeTag, PTAG(Model)) Model;
    typedef typename GET_PROP_TYPE(TypeTag, PTAG(NewtonMethod)) NewtonMethod;
    typedef typename GET_PROP_TYPE(TypeTag, PTAG(JacobianMatrix)) JacobianMatrix;
    typedef typename GET_PROP_TYPE(TypeTag, PTAG(TimeManager)) TimeManager;
    typedef typename GET_PROP_TYPE(TypeTag, PTAG(VertexMapper)) VertexMapper;

    typedef typename GET_PROP_TYPE(TypeTag, PTAG(SolutionVector)) SolutionVector;
    typedef typename GET_PROP_TYPE(TypeTag, PTAG(PrimaryVariables)) PrimaryVariables;

    enum { enableTimeStepRampUp = GET_PROP_VALUE(TypeTag, PTAG(EnableTimeStepRampUp)) };
    enum { enablePartialReassemble = GET_PROP_VALUE(TypeTag, PTAG(EnablePartialReassemble)) };
    enum { enableJacobianRecycling = GET_PROP_VALUE(TypeTag, PTAG(EnableJacobianRecycling)) };
    enum { newtonWriteConvergence = GET_PROP_VALUE(TypeTag, PTAG(NewtonWriteConvergence)) };
    
    typedef NewtonConvergenceWriter<TypeTag, newtonWriteConvergence>  ConvergenceWriter;

/*
    // typedefs for the algebraic overlap
    typedef Dumux::OverlapFromBCRSMatrix<JacobianMatrix> Overlap;
    typedef typename Dumux::VertexBorderListFromGrid<GridView, VertexMapper> BorderListFromGrid;
    typedef typename Overlap::BorderList BorderList;
    typedef typename Dumux::OverlapBCRSMatrix<JacobianMatrix, Overlap> OverlapMatrix;
    typedef typename Dumux::OverlapBlockVector<SolutionVector, Overlap> OverlapVector;
*/

public:
    /*!
     * \brief Constructor
     */
    NewtonController()
        : endIterMsgStream_(std::ostringstream::out),
          convergenceWriter_(asImp_())
    {
        verbose_ = true;
        numSteps_ = 0;

        this->setRelTolerance(1e-8);
        this->rampUpSteps_ = 0;

        if (enableTimeStepRampUp) {
            this->rampUpSteps_ = 9;

            // the ramp-up steps are not counting
            this->setTargetSteps(10);
            this->setMaxSteps(12);
        }
        else {
            this->setTargetSteps(10);
            this->setMaxSteps(18);
        }

        /*
        overlapX_ = NULL;
        overlapB_ = NULL;
        overlapMatrix_ = NULL;
        overlap_ = NULL;
        */
    };

    /*!
     * \brief Destructor
     */
    ~NewtonController()
    {
        /*
        delete overlapX_;
        delete overlapB_;
        delete overlapMatrix_;
        delete overlap_;
        */
    };

    /*!
     * \brief Set the maximum acceptable difference for convergence of
     *        any primary variable between two iterations.
     *
     * \param tolerance The maximum relative error between two Newton
     *                  iterations at which the scheme is considered
     *                  finished
     */
    void setRelTolerance(Scalar tolerance)
    { tolerance_ = tolerance; }

    /*!
     * \brief Set the number of iterations at which the Newton method
     *        should aim at.
     *
     * This is used to control the time step size. The heuristic used
     * is to scale the last time step size by the deviation of the
     * number of iterations used from the target steps.
     *
     * \param targetSteps Number of iterations which are considered "optimal"
     */
    void setTargetSteps(int targetSteps)
    { targetSteps_ = targetSteps; }

    /*!
     * \brief Set the number of iterations after which the Newton
     *        method gives up.
     *
     * \param maxSteps Number of iterations after we give up
     */
    void setMaxSteps(int maxSteps)
    { maxSteps_ = maxSteps; }

    /*!
     * \brief Returns the number of iterations used for the time step
     *        ramp-up.
     */
    Scalar rampUpSteps() const
    { return enableTimeStepRampUp?rampUpSteps_:0; }

    /*!
     * \brief Returns whether the time-step ramp-up is still happening
     */
    bool inRampUp() const
    { return numSteps_ < rampUpSteps(); }

    /*!
     * \brief Returns true if another iteration should be done.
     *
     * \param uCurrentIter The solution of the current newton iteration
     */
    bool newtonProceed(const SolutionVector &uCurrentIter)
    {
        if (numSteps_ < rampUpSteps() + 2)
            return true; // we always do at least two iterations
        else if (asImp_().newtonConverged())
            return false; // we are below the desired tolerance
        else if (numSteps_ >= rampUpSteps() + maxSteps_) {
            // we have exceeded the allowed number of steps.  if the
            // relative error was reduced by a factor of at least 4,
            // we proceed even if we are above the maximum number of
            // steps
            return error_*4.0 < lastError_;
        }

        return true;
    }

    /*!
     * \brief Returns true iff the error of the solution is below the
     *        tolerance.
     */
    bool newtonConverged() const
    {
        return error_ <= tolerance_;
    }

    /*!
     * \brief Called before the newton method is applied to an
     *        non-linear system of equations.
     *
     * \param method The object where the NewtonMethod is executed
     * \param u The initial solution
     */
    void newtonBegin(NewtonMethod &method, const SolutionVector &u)
    {
        method_ = &method;
        numSteps_ = 0;

        dtInitial_ = timeManager_().timeStepSize();
        if (enableTimeStepRampUp) {
            rampUpDelta_ =
                timeManager_().timeStepSize()
                /
                rampUpSteps()
                *
                2;

            // reduce initial time step size for ramp-up.
            timeManager_().setTimeStepSize(rampUpDelta_);
        }

        convergenceWriter_.beginTimestep();
    }

    /*!
     * \brief Indidicates the beginning of a newton iteration.
     */
    void newtonBeginStep()
    { lastError_ = error_; }

    /*!
     * \brief Returns the number of steps done since newtonBegin() was
     *        called.
     */
    int newtonNumSteps()
    { return numSteps_; }

    /*!
     * \brief Update the relative error of the solution compared to
     *        the previous iteration.
     *
     * The relative error can be seen as a norm of the difference
     * between the current and the next iteration.
     *
     * \param uLastIter The current iterative solution
     * \param deltaU The difference between the current and the next solution
     */
    void newtonUpdateRelError(const SolutionVector &uLastIter,
                              const SolutionVector &deltaU)
    {
        // calculate the relative error as the maximum relative
        // deflection in any degree of freedom.
        typedef typename SolutionVector::block_type FV;
        error_ = 0;

        int idxI = -1;
        int aboveTol = 0;
        for (int i = 0; i < int(uLastIter.size()); ++i) {
            PrimaryVariables uNewI = uLastIter[i];
            uNewI -= deltaU[i];
            Scalar vertErr =
                model_().relativeErrorVertex(i,
                                             uLastIter[i],
                                             uNewI);

            if (vertErr > tolerance_)
                ++aboveTol;
            if (vertErr > error_) {
                idxI = i;
                error_ = vertErr;
            }
        }

        if (gridView_().comm().size() > 1)
        	error_ = gridView_().comm().max(error_);
    }

    /*!
     * \brief Solve the linear system of equations \f$\mathbf{A}x - b = 0\f$.
     *
     * Throws Dumux::NumericalProblem if the linear solver didn't
     * converge.
     *
     * \param A The matrix of the linear system of equations
     * \param x The vector which solves the linear system
     * \param b The right hand side of the linear system
     */
    void newtonSolveLinear(JacobianMatrix &A,
                           SolutionVector &x,
                           SolutionVector &b)
    {
        // if the deflection of the newton method is large, we do not
        // need to solve the linear approximation accurately. Assuming
        // that the initial value for the delta vector u is quite
        // close to the final value, a reduction of 9 orders of
        // magnitude in the defect should be sufficient...
        Scalar residReduction = 1e-9;

        try {
            solveLinear_(A, x, b, residReduction);

            // make sure all processes converged
            int converged = 1;
            if (gridView_().comm().size() > 1)
            	gridView_().comm().min(converged);

            if (!converged) {
                DUNE_THROW(NumericalProblem,
                           "A process threw NumericalProblem");
            }
        }
        catch (Dune::MatrixBlockError e) {
            // make sure all processes converged
            int converged = 0;
            if (gridView_().comm().size() > 1)
            	gridView_().comm().min(converged);

            Dumux::NumericalProblem p;
            std::string msg;
            std::ostringstream ms(msg);
            ms << e.what() << "M=" << A[e.r][e.c];
            p.message(ms.str());
            throw p;
        }
        catch (const Dune::ISTLError &e) {
            // make sure all processes converged
            int converged = 0;
            if (gridView_().comm().size() > 1)
            	gridView_().comm().min(converged);

            Dumux::NumericalProblem p;
            p.message(e.what());
            throw p;
        }
    }

    /*!
     * \brief Update the current solution with a delta vector.
     *
     * The error estimates required for the newtonConverged() and
     * newtonProceed() methods should be updated inside this method.
     *
     * Different update strategies, such as line search and chopped
     * updates can be implemented. The default behaviour is just to
     * subtract deltaU from uLastIter, i.e.
     * \f[ u^{k+1} = u^k - \Delta u^k \f]
     *
     * \param uCurrentIter The solution vector after the current iteration
     * \param uLastIter The solution vector after the last iteration
     * \param deltaU The delta as calculated from solving the linear
     *               system of equations. This parameter also stores
     *               the updated solution.
     */
    void newtonUpdate(SolutionVector &uCurrentIter,
                      const SolutionVector &uLastIter,
                      const SolutionVector &deltaU)
    {
        writeConvergence_(uLastIter, deltaU);

        newtonUpdateRelError(uLastIter, deltaU);

        // compute the vertex and element colors for partial
        // reassembly
        if (enablePartialReassemble) {
            Scalar minReasmTol = 0.1*tolerance_;
            Scalar tmp = Dumux::geometricMean(error_, minReasmTol);
            Scalar reassembleTol = Dumux::geometricMean(error_, tmp);
            reassembleTol = std::max(reassembleTol, minReasmTol);
            this->model_().jacobianAssembler().updateDiscrepancy(uLastIter, deltaU);
            this->model_().jacobianAssembler().computeColors(reassembleTol);
        }

        uCurrentIter = uLastIter;
        uCurrentIter -= deltaU;
    }

    /*!
     * \brief Indicates that one newton iteration was finished.
     *
     * \param uCurrentIter The solution after the current Newton iteration
     * \param uLastIter The solution at the beginning of the current Newton iteration
     */
    void newtonEndStep(const SolutionVector &uCurrentIter,
                       const SolutionVector &uLastIter)
    {
        ++numSteps_;

        Scalar realError = error_;
        if (inRampUp() && error_ < 1.0) {
            // change time step size
            Scalar dt = timeManager_().timeStepSize();
            dt += rampUpDelta_;
            timeManager_().setTimeStepSize(dt);

            endIterMsg() << ", dt=" << timeManager_().timeStepSize() << ", ddt=" << rampUpDelta_;
        }

        if (verbose())
            std::cout << "\rNewton iteration " << numSteps_ << " done: "
                      << "error=" << realError << endIterMsg().str() << "\n";
        endIterMsgStream_.str("");
    }

    /*!
     * \brief Indicates that we're done solving the non-linear system
     *        of equations.
     */
    void newtonEnd()
    {
        convergenceWriter_.endTimestep();
    }

    /*!
     * \brief Called if the newton method broke down.
     *
     * This method is called _after_ newtonEnd()
     */
    void newtonFail()
    {
        model_().jacobianAssembler().reassembleAll();
        timeManager_().setTimeStepSize(dtInitial_);
        numSteps_ = targetSteps_*2;
    }

    /*!
     * \brief Called when the newton method was sucessful.
     *
     * This method is called _after_ newtonEnd()
     */
    void newtonSucceed()
    { 
        if (enableJacobianRecycling)
            model_().jacobianAssembler().setMatrixReuseable(true);
        else
            model_().jacobianAssembler().reassembleAll();
    }

    /*!
     * \brief Suggest a new time stepsize based on the old time step
     *        size.
     *
     * The default behaviour is to suggest the old time step size
     * scaled by the ratio between the target iterations and the
     * iterations required to actually solve the last time step.
     */
    Scalar suggestTimeStepSize(Scalar oldTimeStep) const
    {
        if (enableTimeStepRampUp)
            return oldTimeStep;

        Scalar n = numSteps_;
        n -= rampUpSteps();

        // be agressive reducing the timestep size but
        // conservative when increasing it. the rationale is
        // that we want to avoid failing in the next newton
        // iteration which would require another linearization
        // of the problem.
        if (n > targetSteps_) {
            Scalar percent = (n - targetSteps_)/targetSteps_;
            return oldTimeStep/(1.0 + percent);
        }
        else {
            /*Scalar percent = (Scalar(1))/targetSteps_;
              return oldTimeStep*(1 + percent);
            */
            Scalar percent = (targetSteps_ - n)/targetSteps_;
            return std::min(oldTimeStep*(1.0 + percent/1.2), 
                            this->problem_().maxTimeStepSize());
        }
    }

    /*!
     * \brief Returns a reference to the current newton method
     *        which is controlled by this controller.
     */
    NewtonMethod &method()
    { return *method_; }

    /*!
     * \brief Returns a reference to the current newton method
     *        which is controlled by this controller.
     */
    const NewtonMethod &method() const
    { return *method_; }

    std::ostringstream &endIterMsg()
    { return endIterMsgStream_; }

    /*!
     * \brief Specifies if the newton method ought to be chatty.
     */
    void setVerbose(bool val)
    { verbose_ = val; }

    /*!
     * \brief Returns true iff the newton method ought to be chatty.
     */
    bool verbose() const
    { return verbose_ && gridView_().comm().rank() == 0; }

protected:
    /*!
     * \brief Returns a reference to the grid view.
     */
    const GridView &gridView_() const
    { return problem_().gridView(); }

    /*!
     * \brief Returns a reference to the vertex mapper.
     */
    const VertexMapper &vertexMapper_() const
    { return model_().vertexMapper(); }

    /*!
     * \brief Returns a reference to the problem.
     */
    Problem &problem_()
    { return method_->problem(); }

    /*!
     * \brief Returns a reference to the problem.
     */
    const Problem &problem_() const
    { return method_->problem(); }

    /*!
     * \brief Returns a reference to the time manager.
     */
    TimeManager &timeManager_()
    { return problem_().timeManager(); }

    /*!
     * \brief Returns a reference to the time manager.
     */
    const TimeManager &timeManager_() const
    { return problem_().timeManager(); }

    /*!
     * \brief Returns a reference to the problem.
     */
    Model &model_()
    { return problem_().model(); }

    /*!
     * \brief Returns a reference to the problem.
     */
    const Model &model_() const
    { return problem_().model(); }

    // returns the actual implementation for the cotroller we do
    // it this way in order to allow "poor man's virtual methods",
    // i.e. methods of subclasses which can be called by the base
    // class.
    Implementation &asImp_()
    { return *static_cast<Implementation*>(this); }
    const Implementation &asImp_() const
    { return *static_cast<const Implementation*>(this); }

    void writeConvergence_(const SolutionVector &uLastIter,
                           const SolutionVector &deltaU)
    {
        convergenceWriter_.beginIteration(this->gridView_());
        convergenceWriter_.writeFields(uLastIter, deltaU);
        convergenceWriter_.endIteration();
    };

    /*!
     * \brief Actually invoke the linear solver
     *
     * Usually we use the solvers from DUNE-ISTL.
     */
    void solveLinear_(JacobianMatrix &A,
                      SolutionVector &x,
                      SolutionVector &b,
                      Scalar residReduction)
    {
        typedef typename GET_PROP_TYPE(TypeTag, PTAG(GridView)) GridView;
        typedef typename GET_PROP_TYPE(TypeTag, PTAG(VertexMapper)) VertexMapper;
        VertexBorderListFromGrid<GridView, VertexMapper>
            borderListCreator(problem_().gridView(), problem_().vertexMapper());

        typedef Dumux::OverlappingBCRSMatrix<JacobianMatrix> OverlappingMatrix;
        typedef typename OverlappingMatrix::Overlap Overlap;
        typedef Dumux::OverlappingBlockVector<typename SolutionVector::block_type, Overlap> OverlappingVector;

        OverlappingMatrix overlapA(A, 
                                   borderListCreator.borderList(), 
                                   /*overlapSize=*/4);
        overlapA.print();

        OverlappingVector overlapb(b, overlapA.overlap());
        overlapb.print();

        /*
        typedef Dumux::OverlapBCRSMatrix<JacobianMatrix, GlobalIndices> OverlapMatrix;
        OverlapMatrix overlapA(A, globalIndices);
        */

        DUNE_THROW(Dune::NotImplemented,
                   "NewtonController::solveLinear_");
#if 0
        int verbosity = GET_PROP_VALUE(TypeTag, PTAG(NewtonLinearSolverVerbosity));
        int myRank = gridView_().comm().rank();
        if (myRank != 0)
            verbosity = 0;

        updateOverlap_(A, x, b);
        //printmatrix(std::cout, *((JacobianMatrix *) overlapMatrix_), "M", "row");

#define PREC 3
#if PREC == 1
        // simple Jacobi preconditioner
        typedef Dune::SeqJac<OverlapMatrix, OverlapVector, OverlapVector> SeqPreconditioner;
        SeqPreconditioner seqPreCond(*overlapMatrix_, 1, 1.0);
#elif PREC == 2
        // SSOR preconditioner
        typedef Dune::SeqSSOR<OverlapMatrix, OverlapVector, OverlapVector> SeqPreconditioner;
        SeqPreconditioner seqPreCond(*overlapMatrix_, 1, 1.0);
#elif PREC == 3
        // ILU preconditioner
        typedef Dune::SeqILU0<OverlapMatrix, OverlapVector, OverlapVector> SeqPreconditioner;
        SeqPreconditioner seqPreCond(*overlapMatrix_, 1.0);
#endif

        typedef Dumux::OverlapPreconditioner<SeqPreconditioner, OverlapMatrix, Overlap> OverlapPreconditioner;
        OverlapPreconditioner preCond(seqPreCond, *overlap_);
        
        typedef Dumux::OverlapOperator<OverlapMatrix,OverlapVector,OverlapVector> LinearOperator;
        //typedef Dune::MatrixAdapter<OverlapMatrix,OverlapVector,OverlapVector> LinearOperator;
        LinearOperator opA(*overlapMatrix_);

        typedef Dumux::OverlapScalarProduct<OverlapVector, Overlap> OverlapScalarProduct;
        OverlapScalarProduct scalarProd(*overlap_);

        typedef Dune::BiCGSTABSolver<OverlapVector> Solver;
        Solver solver(opA, 
                      scalarProd,
                      preCond, 
                      residReduction,
                      250,
                      1);
    
#if 0
        OverlapVector tmp(*overlapB_);

        //overlapMatrix_->mv(*overlapB_, tmp);
        //overlapMatrix_->mv(tmp, *overlapB_);
        //overlap_->printOverlap();

        std::cout << "before precond: result: " << tmp << " rhs: " << *overlapB_ << "\n";
        preCond.pre(tmp, *overlapB_);
        preCond.apply(tmp, *overlapB_);
        preCond.post(tmp);
        std::cout << "after precond: result: " << tmp << " rhs: " << *overlapB_ << "\n";

        exit(1);
      
        Scalar dot = scalarProd.dot(tmp, tmp);
        Scalar norm = scalarProd.norm(tmp);
        std::cout.precision(16);
        std::cout << "num domestic:" << overlap_->numDomestic() << "\n";
        std::cout << "Dot:" << dot << "\n";
        std::cout << "norm^2:" << norm*norm << "\n";
        exit(1);
#endif
    
//        typedef Dune::RestartedGMResSolver<OverlapVector> Solver;
//        Solver solver(opA, scalarProd, precond, residReduction, 50, 500, verbosity);

        /*
        std::vector<int> per(overlapX_->size());
        assert(overlapX_->size() == 13*9);
        if (mySize == 2) {
            for (int i = 0; i < 7; ++i)
                for (int j = 0; j < 9; ++j)
                    per[j*7 + i] = j*13 + i;
            for (int i = 7; i < 13; ++i)
                for (int j = 0; j < 9; ++j)
                    per[j*(13 - 7) + (i - 7) + 7*9] = j*13 + i;
        }
        else {
            for (int i = 0; i < overlapX_->size(); ++i)
                per[i] = i;
        }
        
        std::ofstream *lfp;
        if (myRank == 0) {
            lfp = new std::ofstream("/tmp/overlap.log");
            std::ofstream &lf = *lfp;
            lf.precision(18);
            typedef typename OverlapMatrix::ColIterator ColIt;
            for (int row = 0; row < overlapMatrix_->N(); ++ row) {
                ColIt it = (*overlapMatrix_)[row].begin();
                ColIt endIt = (*overlapMatrix_)[row].end();
                for (; it != endIt; ++it) { 
                    lf << "row "  << per[row]
                       << " col " << per[it.index()]
                       << " = "
                       << (*it)[0][0] << " "
                       << (*it)[0][1]
                       << " / " 
                       << (*it)[1][0] << " "
                       << (*it)[1][1]
                       << "\n";
                }
            }
            
            SolutionVector tmp(overlapX_->size());
            for (int i = 0; i < overlapX_->size(); ++i) {
                lf << "vec b row "  << per[i]
                   << "      = " << (*overlapB_)[i]
                   << "\n";
                lf << "vec x init row "  << per[i]
                   << "      = " << (*overlapX_)[i]
                   << "\n";
            }
        }
        */
        
        Dune::InverseOperatorResult result;
        solver.apply(*overlapX_, *overlapB_, result);
        
        overlapX_->sync();
        if (!result.converged)
            DUNE_THROW(Dumux::NumericalProblem,
                       "Solving the linear system of equations did not converge.");
        /*
        if (myRank == 0) {
            std::ofstream &lf = *lfp;
            
            for (int i = 0; i < overlapX_->size(); ++i) {
                lf << "vec b final row "  << per[i]
                   << "      = " << (*overlapB_)[i]
                   << "\n";
                lf << "vec x final row "  << per[i]
                   << "      = " << (*overlapX_)[i]
                   << "\n";
            }
            lf.close();
            delete lfp;
        }
        exit(1);
        */

        // make sure the solver didn't produce a nan or an inf
        // somewhere. this should never happen but for some strange
        // reason it happens anyway.
       
        //Scalar xNorm2 = overlapX_->two_norm2();
        Scalar xNorm2 = x.two_norm2();
        gridView_().comm().sum(xNorm2);
        if (std::isnan(xNorm2) || !std::isfinite(xNorm2))
            DUNE_THROW(Dumux::NumericalProblem,
                       "The linear solver produced a NaN or inf somewhere.");
       
        // copy the result back to the non-overlapping vector
        overlapX_->assignToNonOverlapping(x);
#endif
    }

    void updateOverlap_(const JacobianMatrix &A,
                        const SolutionVector &x,
                        const SolutionVector &b)
    {
        DUNE_THROW(Dune::NotImplemented,
                   "NewtonController::updateOverlap_");
        
#if 0
        if (!overlap_) {
            // create overlap
#warning HACK: only valid for vertex-based methods
            typedef typename Dumux::VertexBorderListFromGrid<GridView, VertexMapper> BorderListFromGrid;

            BorderListFromGrid borderListCreator(gridView_(), vertexMapper_());
#warning HACK: make this a property!
            int overlapSize = 100;
            overlap_ = new Overlap(A,
                                   borderListCreator.borderList(),
                                   overlapSize);
            
            overlapMatrix_ = new OverlapMatrix(A, *overlap_);
            overlapX_ = new OverlapVector(x, *overlap_);
            overlapB_ = new OverlapVector(b, *overlap_);
        }
        /*for (int i = 0; i < x.size(); ++i) {
            (*overlapX_)[i] = x[i];
            (*overlapB_)[i] = b[i];
        }
        */
        overlapB_->set(b);
        overlapX_->set(x);

        delete overlapMatrix_;
        overlapMatrix_ = new OverlapMatrix(A, *overlap_);
        overlapMatrix_->set(A);
#endif
    };

    std::ostringstream endIterMsgStream_;

    NewtonMethod *method_;
    bool verbose_;

    ConvergenceWriter convergenceWriter_;

    Scalar error_;
    Scalar lastError_;
    Scalar tolerance_;

    // number of iterations for the time-step ramp-up
    Scalar rampUpSteps_;
    // the increase of the time step size during the rampup
    Scalar rampUpDelta_;

    Scalar dtInitial_; // initial time step size

    // optimal number of iterations we want to achive
    int targetSteps_;
    // maximum number of iterations we do before giving up
    int maxSteps_;
    // actual number of steps done so far
    int numSteps_;

    // objects for the algebraic overlap
    //Overlap *overlap_;
    //OverlapMatrix *overlapMatrix_;
    //OverlapVector *overlapX_;
    //OverlapVector *overlapB_;
};
} // namespace Dumux

#endif
