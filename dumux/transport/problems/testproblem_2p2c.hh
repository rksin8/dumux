// $Id$

#ifndef TESTPROBLEM_2P2C_HH
#define TESTPROBLEM_2P2C_HH

#include "dumux/transport/transportproblem2p2c.hh"

namespace Dune
{

//! Example problem class for decoupled 2p2c simulations
template<class Grid, class Scalar>
class Testproblem_2p2c
    : public TransportProblem2p2c<Grid, Scalar>
{
    template<int dim>
    struct ElementLayout
    {
        bool contains (GeometryType gt)
        {
            return gt.dim() == dim;
        }
    };

    enum {dim=Grid::dimension};
    typedef typename Grid::Traits::template Codim<0>::Entity Entity;

public:

    virtual const FieldVector<Scalar,dim> gravity()
    {
        FieldVector<Scalar,dim> gravity_(0);
        gravity_[2] = -10;
        return gravity_;
    }

    Scalar temp(const FieldVector<Scalar, dim>& globalPos, const Entity& element,
                                           const FieldVector<Scalar, dim>& localPos)
    {
        return 283.15;
    }

    BoundaryConditions2p2c::Flags bc_type (const FieldVector<Scalar, dim>& globalPos, const Entity& element,
                                           const FieldVector<Scalar, dim>& localPos) const
    {

        return BoundaryConditions2p2c::concentration;
    }

    BoundaryConditions2p2c::Flags initcond_type (const FieldVector<Scalar, dim>& globalPos, const Entity& element,
                                                 const FieldVector<Scalar, dim>& localPos) const
    {
        return BoundaryConditions2p2c::concentration;
    }

    BoundaryConditions::Flags press_bc_type (const Dune::FieldVector<Scalar, dim>& globalPos, const Entity& element,
                                             const Dune::FieldVector<Scalar, dim>& localPos) const
    {
        if (globalPos[0] > 10-1E-6 || globalPos[0] < 1e-6)
            return Dune::BoundaryConditions::dirichlet;
        // all other boundaries
        return Dune::BoundaryConditions::neumann;
    }


    Scalar dirichlet (const FieldVector<Scalar, dim>& globalPos, const Entity& element, const FieldVector<Scalar, dim>& localPos) const
    {
        return (globalPos[0] < 1e-6) ? (2.5e5 - 10000 * globalPos[2]) : (2e5 - 10000 * globalPos[2]);
    }

    Scalar dirichletConcentration (const FieldVector<Scalar, dim>& globalPos, const Entity& element,
                                   const FieldVector<Scalar, dim>& localPos) const
    {
        return 1;
    }

    Scalar dirichletSat (const FieldVector<Scalar, dim>& globalPos, const Entity& element,
                         const FieldVector<Scalar, dim>& localPos) const
    {
        if (globalPos[0] < 15)
            return 0;
        else
            return 0;
    }

    virtual FieldVector<Scalar,2> neumann (const FieldVector<Scalar, dim>& globalPos, const Entity& element,
                                           const FieldVector<Scalar, dim>& localPos) const
    {
        FieldVector<Scalar,2> J_(0);
        return J_;
    }

    virtual FieldVector<Scalar,2> source (const FieldVector<Scalar, dim>& globalPos, const Entity& element,
                                          const FieldVector<Scalar, dim>& localPos) const
    {
        FieldVector<Scalar,2> q_(0);
        if (fabs(globalPos[0] - 4.5) < 1 && fabs(globalPos[1] - 4.5) < 1) q_[1] = 0.0001;
        return q_;
    }

    Scalar initSat (const FieldVector<Scalar, dim>& globalPos, const Entity& element,
                    const FieldVector<Scalar, dim>& localPos) const
    {
        return 0.999;
    }

    Scalar initConcentration(const FieldVector<Scalar, dim>& globalPos, const Entity& element,
                             const FieldVector<Scalar, dim>& localPos) const
    {
        return 1;
    }

    Testproblem_2p2c(Grid& g, Dune::VariableClass2p2c<Grid, Scalar>& var, Liquid_GL& liq, Gas_GL& gas, Matrix2p<Grid, Scalar>& s,
                     int level, TwoPhaseRelations<Grid, Scalar>& law = *(new TwoPhaseRelations<Grid, Scalar>),const bool cap = false)
        : TransportProblem2p2c<Grid, Scalar>(var, liq, gas, s, law, cap), grid(g)
    {
    }

private:
    Grid& grid;
};

}
#endif
