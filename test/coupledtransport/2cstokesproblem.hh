#ifndef DUNE_TWOCSTOKESPROBLEM_HH
#define DUNE_TWOCSTOKESPROBLEM_HH

#include"dumux/stokes/stokestransportproblem.hh"

namespace Dune {

template<class Grid, class Scalar>
class TwoCStokesProblem : public StokesTransportProblem<Grid, Scalar>
{
    enum {velocityXIdx=0, velocityYIdx=1, partialDensityIdx=2, pressureIdx=3};
    enum {dim=Grid::dimension, numEq=Grid::dimension+2};
    typedef typename Grid::Traits::template Codim<0>::Entity Element;
    typedef typename Grid::LeafGridView::IntersectionIterator IntersectionIterator;
    typedef FieldVector<Scalar,dim> GlobalPosition;
    typedef FieldVector<Scalar,dim> LocalPosition;
    typedef FieldVector<Scalar,numEq> SolutionVector;
public:


    SolutionVector initial (const GlobalPosition& globalPos, const Element& element,
                            const LocalPosition& localPos) const
    {

	SolutionVector result(0);

        result[velocityXIdx] = 1.0e-3;
        result[velocityYIdx] = 0;
        result[partialDensityIdx] = 0.1;
        result[pressureIdx] = 1e5;

        return result;
    }

    SolutionVector q(const GlobalPosition& globalPos, const Element& element,
                     const LocalPosition& localPos) const
    {
        SolutionVector result(0);

        return result;
    }

    virtual FieldVector<BoundaryConditions::Flags, numEq> bctype (const GlobalPosition& globalPos, const Element& element,
                                      const IntersectionIterator& intersectionIt,
                                      const LocalPosition& localPos) const
    {
    	FieldVector<BoundaryConditions::Flags, numEq> values(BoundaryConditions::dirichlet);

    	if (globalPos[0] > eps_ || globalPos[1] > eps_ || globalPos[1] < 1 - eps_ || globalPos[0] < 5.5 - eps_)
    	{
            values[0] = BoundaryConditions::neumann;
            values[1] = BoundaryConditions::neumann;
            values[2] = BoundaryConditions::neumann;
            values[3] = BoundaryConditions::neumann;
    	}

    	return values;
    }

    SolutionVector dirichlet(const GlobalPosition& globalPos, const Element& element,
                     const IntersectionIterator& intersectionIt,
                     const LocalPosition& localPos) const
    {
        SolutionVector result(0);

        result[velocityXIdx] = velocity(globalPos, element, localPos)[0];
        result[velocityYIdx] = velocity(globalPos, element, localPos)[1];
        result[partialDensityIdx] = 0.1;
//        result[pressureIdx] = 1e5;

        return result;
    }

    SolutionVector neumann(const GlobalPosition& globalPos, const Element& element,
                     const IntersectionIterator& intersectionIt,
                     const LocalPosition& localPos)
    {
        SolutionVector result(0);

        return result;
    }

    // function returns the square root of the permeability (scalar) divided by alpha times mu
    virtual Scalar beaversJosephC(const GlobalPosition& globalPos, const Element& element,
                                  const IntersectionIterator& intersectionIt,
                                  const LocalPosition& localPos) const
    {
        Scalar alpha;
        // tangential face of porous media
        if (globalPos[0] < 4.0 + eps_ && globalPos[1] < 0.5 + eps_)
            alpha = 0.1;

        else // right boundary
            return(-1.0); // realizes outflow boundary condition

        //TODO: uses only Kxx, extend to permeability tensor
        Scalar permeability = soil().K(globalPos, element, localPos)[0][0];

        return sqrt(permeability)/(alpha*viscosity(globalPos, element, localPos));
    }

    //TODO: call viscosity from material law
    virtual Scalar viscosity(const GlobalPosition& globalPos, const Element& element, const LocalPosition& localPos) const
    {
        return 0.01;
    }

    virtual SolutionVector velocity(const GlobalPosition& globalPos, const Element& element,
                                    const LocalPosition& localPos) const
    {
        SolutionVector result(0);

        result[velocityXIdx] = 4.0*globalPos[1]*(1.0 - globalPos[1]);

        return result;
    }

    FieldMatrix<Scalar,dim,dim> D (const GlobalPosition& globalPos, const Element& element,
                                   const LocalPosition& localPos) const
    {
        FieldMatrix<Scalar,dim,dim> res(0);

        for (int Dx=0; Dx<dim; Dx++)
            for (int Dy=0; Dy<dim; Dy++)
                if (Dx == Dy)
                    res[Dx][Dy] = 1e-5;

        return res;
    }

    Scalar Qg(const GlobalPosition& globalPos, const Element& element,
                      const LocalPosition& localPos) const
    {
        Scalar result = 0;
        return result;
    }

    Fluid& gasPhase () const
    {
        return gasPhase_;
    }

    MultiComp& multicomp () const
    {
        return multicomp_;
    }

    //TODO: gravity vector instead of scalar
    //    FieldVector<Scalar,dim>& gravity() const
    Scalar gravity () const
    {
        return gravity_;
    }

    Matrix2p<Grid, Scalar>& soil() const
    {
        return soil_;
    }
    /*
      FieldMatrix<Scalar, dim, dim> velocityGradient(const GlobalPosition& globalPos) const
      {
      FieldMatrix<Scalar, dim, dim> result(0);

      return result;
      }
    */

    TwoCStokesProblem(Gas_GL& gasPhase, Matrix2p<Grid, Scalar>& soil, MultiComp& multicomp = *(new CWaterAir))
        :
        StokesTransportProblem<Grid,Scalar>(gasPhase, multicomp),
        gasPhase_(gasPhase),
        soil_(soil),
        multicomp_(multicomp)
    {
        gravity_ = 0;//9.81;
        eps_ = 1e-6;
        //    for (int i=0; i<dim; ++i)
        //        gravity_[i] = 0;
        //    gravity_[dim] = -9.81;
    }


protected:
    //    FieldVector<Scalar,dim> gravity_;
    Scalar gravity_;
    Gas_GL& gasPhase_;
    Matrix2p<Grid, Scalar>& soil_;
    MultiComp& multicomp_;
    Scalar eps_;
};

}
#endif
