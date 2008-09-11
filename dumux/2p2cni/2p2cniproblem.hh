#ifndef DUNE_TWOPTWOCNIPROBLEM_HH
#define DUNE_TWOPTWOCNIPROBLEM_HH

#include<iostream>
#include<iomanip>

#include<dune/common/fvector.hh>
#include<dune/common/fmatrix.hh>
#include<dune/common/exceptions.hh>
#include<dune/grid/common/grid.hh>
#include<dune/grid/common/referenceelements.hh>
#include<dune/grid/utility/intersectiongetter.hh>
#include<dune/disc/operators/boundaryconditions.hh>
#include<dumux/material/twophaserelations.hh>
#include<dumux/material/multicomponentrelations.hh>
#include<dumux/material/property_baseclasses.hh>


/**
 * @file
 * @brief  Base class for defining an instance of the non-isothermal two-phase two-component problem
 * @author Bernd Flemisch
 */

namespace Dune {
//! base class that defines the parameters of a diffusion equation
/*! An interface for defining parameters for the stationary diffusion equation
 * \f$ - \text{div}\, (\lambda K \text{grad}\, p ) = q, \f$,
 * \f$p = g\f$ on \f$\Gamma_1\f$, and \f$\lambda K \text{grad}\, p = J\f$
 * on \f$\Gamma_2\f$. Here,
 * \f$p\f$ denotes the pressure, \f$K\f$ the absolute permeability,
 * and \f$\lambda\f$ the total mobility, possibly depending on the
 * saturation.
 *
 *	Template parameters are:
 *
 *	- Grid  a DUNE grid type
 *	- RT    type used for return values
 */
template<class G, class RT> class TwoPTwoCNIProblem {
	typedef typename G::ctype DT;
	enum {dim=G::dimension, m=3};
	typedef typename G::Traits::template Codim<0>::Entity Entity;
	typedef typename IntersectionIteratorGetter<G,LeafTag>::IntersectionIterator
			IntersectionIterator;

public:
	//! evaluate diffusion tensor
	/*! Evaluate the diffusion tensor at given location
	  @param[in]  x    position in global coordinates
	  @param[in]  e    entity of codim 0
	  @param[in]  xi   position in reference element of e
	  @param[out] D    diffusion tensor to be filled
	 */
	virtual const FieldVector<DT,2>& D (const FieldVector<DT,dim>& x) const = 0;

	//! evaluate source term
	/*! evaluate source term at given location
	 @param[in]  x    position in global coordinates
	 @param[in]  e    entity of codim 0
	 @param[in]  xi   position in reference element of e
	 \return     value of source term
	 */
	virtual FieldVector<RT,m> q(const FieldVector<DT,dim>& x, const Entity& e,
			const FieldVector<DT,dim>& xi) const = 0;

	//! return type of boundary condition at the given global coordinate
	/*! return type of boundary condition at the given global coordinate
	 @param[in]  x    position in global coordinates
	 \return     boundary condition type given by enum in this class
	 */
	//	virtual FieldVector<BoundaryConditions::Flags, m> bctype (const FieldVector<DT,dim>& x, const Entity& e,
	//			const IntersectionIterator& intersectionIt,
	//			const FieldVector<DT,dim>& xi) const = 0;

	virtual FieldVector<BoundaryConditions::Flags, m>bctype(
			const FieldVector<DT,dim>& x, const Entity& e,
			const IntersectionIterator& intersectionIt,
			const FieldVector<DT,dim>& xi) const = 0;

	//! returns index of the primary variable corresponding to the dirichlet boundary condition at the given global coordinate
		/*! returns index of the primary variable corresponding to the dirichlet boundary condition at the given global coordinate
		 @param[in]  x    position in global coordinates
		 \return     index of the primary variable
		 */

	virtual void dirichletIndex(const FieldVector<DT,dim>& x, const Entity& e,
			const IntersectionIterator& intersectionIt,
			const FieldVector<DT,dim>& xi, FieldVector<int,m>& dirichletIndex) const
	{
		for (int i = 0; i < m; i++)
			dirichletIndex[i]=i;
		return;
	}

	//! evaluate Dirichlet boundary condition at given position
	/*! evaluate Dirichlet boundary condition at given position
	 @param[in]  x    position in global coordinates
	 \return     boundary condition value
	 */
	virtual FieldVector<RT,m> g(const FieldVector<DT,dim>& x, const Entity& e,
			const IntersectionIterator& intersectionIt,
			const FieldVector<DT,dim>& xi) const = 0;

	//! evaluate Neumann boundary condition at given position
	/*! evaluate Neumann boundary condition at given position
	 @param[in]  x    position in global coordinates
	 \return     boundary condition value
	 */
	virtual FieldVector<RT,m> J(const FieldVector<DT,dim>& x, const Entity& e,
			const IntersectionIterator& intersectionIt,
			const FieldVector<DT,dim>& xi) const = 0;

	//! evaluate initial condition at given position
	/*! evaluate initial boundary condition at given position
	 @param[in]  x    position in global coordinates
	 \return     boundary condition value
	 */
	virtual FieldVector<RT,m> initial(const FieldVector<DT,dim>& x,
			const Entity& e, const FieldVector<DT,dim>& xi) const = 0;

	//! initiate phase state at given position
	/*! initiate phase state at given position
	  @param[in]  x    position in global coordinates
	  \return     initial phase state
	 */
	virtual int initialPhaseState(const FieldVector<DT,dim>& x,
			const Entity& e, const FieldVector<DT,dim>& xi) const = 0;


	virtual FieldVector<RT,dim> gravity() const = 0;

	virtual MultiComp& multicomp ()
	{
		return multicomp_;
	}

	virtual Liquid_GL& wettingPhase () const = 0;

	virtual Gas_GL& nonwettingPhase () const = 0;

	virtual TwoPhaseRelations<G, RT>& materialLaw () const = 0;

    virtual Matrix2p<G, RT>& soil () const = 0;

	//element-wise return of the values of an Exact solution
	virtual RT uExOutVertex(int &ElementIndex, int VariableIndex) const {
		DUNE_THROW(NotImplemented, "Ex(akt) Solution");
		return 0;
	}

	//updates an exact/analytic solution
	virtual void updateExSol(double &dt,
			BlockVector<FieldVector<RT, m> > &approxSol) {
		DUNE_THROW(NotImplemented, "Ex(akt) Solution");
		return;
	}

	TwoPTwoCNIProblem(Matrix2p<G,RT>& soil, MultiComp& multicomp = *(new CWaterAir),
		TwoPhaseRelations<G,RT>& materialLaw = *(new TwoPhaseRelations<G,RT>), const bool exsol = false) :
	  exsolution(exsol),  materialLaw_(materialLaw) , multicomp_(multicomp), soil_(soil)
	  {	}

	//! always define virtual destructor in abstract base class
	virtual ~TwoPTwoCNIProblem() {
	}

	const bool exsolution;

protected:
	TwoPhaseRelations<G,RT>& materialLaw_;
	MultiComp& multicomp_;
    Matrix2p<G, RT>& soil_;
};

}
#endif
