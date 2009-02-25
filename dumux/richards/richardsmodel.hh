// $Id: twophasemodel.hh 716 2008-10-21 10:00:09Z lauser $

#ifndef DUNE_TWOPHASEMODEL_HH
#define DUNE_TWOPHASEMODEL_HH

#include <dune/disc/shapefunctions/lagrangeshapefunctions.hh>
#include <dune/disc/operators/p1operator.hh>
#include <dune/disc/functions/p1function.hh>
#include "dumux/nonlinear/nonlinearmodel.hh"
#include "dumux/fvgeometry/fvelementgeometry.hh"

namespace Dune {
template<class G, class RT, class ProblemType, class LocalJacobian,
		class FunctionType, class OperatorAssembler> class RichardsModel :
	public NonlinearModel<G, RT, ProblemType, LocalJacobian, FunctionType, OperatorAssembler> {
public:
	typedef NonlinearModel<G, RT, ProblemType, LocalJacobian,
	FunctionType, OperatorAssembler> ThisNonlinearModel;

	RichardsModel(const G& g, ProblemType& prob) :
		ThisNonlinearModel(g, prob), uOldTimeStep(g, g.overlapSize(0)==0) {
	}

	RichardsModel(const G& g, ProblemType& prob, int level) :
		ThisNonlinearModel(g, prob, level), uOldTimeStep(g, level, g.overlapSize(0)==0) {
	}

	virtual void initial() = 0;

	virtual void update(double& dt) = 0;

	virtual void solve() = 0;

	FunctionType uOldTimeStep;
};

template<class G, class RT, class ProblemType, class LocalJac, int m=1>
class LeafP1TwoPhaseModel
: public RichardsModel<G, RT, ProblemType, LocalJac,
                       LeafP1Function<G, RT, m>,
                       LeafP1OperatorAssembler<G, RT, m> >
{
public:
	// define the function type:
	typedef LeafP1Function<G, RT, m> FunctionType;

	// define the operator assembler type:
	typedef LeafP1OperatorAssembler<G, RT, m> OperatorAssembler;

	typedef RichardsModel<G, RT, ProblemType, LocalJac,
	FunctionType, OperatorAssembler> ThisRichardsModel;

	typedef LeafP1TwoPhaseModel<G, RT, ProblemType, LocalJac, m> ThisType;

	typedef LocalJac LocalJacobian;

	// mapper: one data element per vertex
	template<int dim> struct P1Layout {
		bool contains(Dune::GeometryType gt) {
			return gt.dim() == 0;
		}
	};

	typedef typename G::LeafGridView GV;
    typedef typename GV::IndexSet IS;
	typedef MultipleCodimMultipleGeomTypeMapper<G,IS,P1Layout> VertexMapper;
	typedef typename IntersectionIteratorGetter<G,LeafTag>::IntersectionIterator
			IntersectionIterator;

	LeafP1TwoPhaseModel(const G& g, ProblemType& prob) :
		ThisRichardsModel(g, prob), problem(prob), grid_(g), vertexmapper(g,
				g.leafIndexSet()), size((*(this->u)).size()), pW(size), pC(size),
				satW(size), satEx(0), pEx(0), satError(0) {
	}

	virtual void initial() {
		typedef typename G::Traits::template Codim<0>::Entity Entity;
		typedef typename G::ctype DT;
		typedef typename GV::template Codim<0>::Iterator Iterator;
		enum {dim = G::dimension};
		enum {dimworld = G::dimensionworld};

		const GV& gridview(this->grid_.leafView());

		// iterate through leaf grid an evaluate c0 at cell center
		Iterator eendit = gridview.template end<0>();
		for (Iterator it = gridview.template begin<0>(); it
				!= eendit; ++it) {
			// get geometry type
			Dune::GeometryType gt = it->geometry().type();

			// get entity
			const Entity& entity = *it;

			const typename Dune::LagrangeShapeFunctionSetContainer<DT,RT,dim>::value_type
					&sfs=Dune::LagrangeShapeFunctions<DT, RT, dim>::general(gt,
							1);
			int size = sfs.size();

			for (int i = 0; i < size; i++) {
				// get cell center in reference element
				const Dune::FieldVector<DT,dim>&local = sfs[i].position();

				// get global coordinate of cell center
				Dune::FieldVector<DT,dimworld> global = it->geometry().global(local);

				int globalId = vertexmapper.template map<dim>(entity,
						sfs[i].entity());

				// initialize cell concentration
				(*(this->u))[globalId] = this->problem.initial(
						global, entity, local);
			}
		}

		// set Dirichlet boundary conditions
		for (Iterator it = gridview.template begin<0>(); it
				!= eendit; ++it) {
			// get geometry type
			Dune::GeometryType gt = it->geometry().type();

			// get entity
			const Entity& entity = *it;

			const typename Dune::LagrangeShapeFunctionSetContainer<DT,RT,dim>::value_type
					&sfs=Dune::LagrangeShapeFunctions<DT, RT, dim>::general(gt,
							1);
			int size = sfs.size();

			// set type of boundary conditions
			this->localJacobian().template assembleBC<LeafTag>(entity);

			IntersectionIterator
					endit = IntersectionIteratorGetter<G, LeafTag>::end(entity);
			for (IntersectionIterator is = IntersectionIteratorGetter<G,
					LeafTag>::begin(entity); is!=endit; ++is)
				if (is->boundary()) {
					for (int i = 0; i < size; i++)
						// handle subentities of this face
						for (int j = 0; j < ReferenceElements<DT,dim>::general(gt).size(is->numberInSelf(), 1, sfs[i].codim()); j++)
							if (sfs[i].entity()
									== ReferenceElements<DT,dim>::general(gt).subEntity(is->numberInSelf(), 1,
											j, sfs[i].codim())) {
								for (int equationNumber = 0; equationNumber<m; equationNumber++) {
									if (this->localJacobian().bc(i)[equationNumber]
											== BoundaryConditions::dirichlet) {
										// get cell center in reference element
										Dune::FieldVector<DT,dim>
												local = sfs[i].position();

										// get global coordinate of cell center
										Dune::FieldVector<DT,dimworld>
												global = it->geometry().global(local);

										int
												globalId = vertexmapper.template map<dim>(
														entity, sfs[i].entity());
										FieldVector<int,m> dirichletIndex;
										FieldVector<BoundaryConditions::Flags, m>
												bctype = this->problem.bctype(
														global, entity, is,
														local);
												this->problem.dirichletIndex(global, entity, is,
														local, dirichletIndex);

										if (bctype[equationNumber]
												== BoundaryConditions::dirichlet) {
											FieldVector<RT,m>
													ghelp = this->problem.g(
															global, entity, is,
															local);
											(*(this->u))[globalId][dirichletIndex[equationNumber]]
													= ghelp[dirichletIndex[equationNumber]];
										}
									}
								}
							}
				}
		}

		*(this->uOldTimeStep) = *(this->u);
		return;
	}

	virtual void update(double& dt) {
		this->localJacobian().setDt(dt);
		this->localJacobian().setOldSolution(this->uOldTimeStep);
		NewtonMethod<G, ThisType> newtonMethod(this->grid_, *this);
		newtonMethod.execute();
		dt = this->localJacobian().getDt();
		*(this->uOldTimeStep) = *(this->u);

		if (this->problem.exsolution)
			this->problem.updateExSol(dt, *(this->u));

		return;
	}

	virtual void globalDefect(FunctionType& defectGlobal) {
		typedef typename G::Traits::template Codim<0>::Entity Entity;
		typedef typename G::ctype DT;
		typedef typename GV::template Codim<0>::Iterator Iterator;
		enum {dim = G::dimension};
		typedef array<BoundaryConditions::Flags, m> BCBlockType;

		const GV& gridview(this->grid_.leafView());
		(*defectGlobal)=0;

		// allocate flag vector to hold flags for essential boundary conditions
		std::vector<BCBlockType> essential(this->vertexmapper.size());
		for (typename std::vector<BCBlockType>::size_type i=0; i
				<essential.size(); i++)
			essential[i].assign(BoundaryConditions::neumann);

		// iterate through leaf grid
		Iterator eendit = gridview.template end<0>();
		for (Iterator it = gridview.template begin<0>(); it
				!= eendit; ++it) {
			// get geometry type
			Dune::GeometryType gt = it->geometry().type();

			// get entity
			const Entity& entity = *it;
			this->localJacobian().fvGeom.update(entity);
			int size = this->localJacobian().fvGeom.numVertices;

			this->localJacobian().setLocalSolution(entity);
			this->localJacobian().computeElementData(entity);
			bool old = true;
			this->localJacobian().updateVariableData(entity, this->localJacobian().uold, old);
			this->localJacobian().updateVariableData(entity, this->localJacobian().u);
			this->localJacobian().template localDefect<LeafTag>(entity, this->localJacobian().u);

			// begin loop over vertices
			for (int i=0; i < size; i++) {
				int globalId = this->vertexmapper.template map<dim>(entity,i);
				for (int equationnumber = 0; equationnumber < m; equationnumber++) {
					if (this->localJacobian().bc(i)[equationnumber] == BoundaryConditions::neumann)
						(*defectGlobal)[globalId][equationnumber]
								+= this->localJacobian().def[i][equationnumber];
					else
						essential[globalId].assign(BoundaryConditions::dirichlet);
				}
			}
		}

		for (typename std::vector<BCBlockType>::size_type i=0; i
				<essential.size(); i++)
			for (int equationnumber = 0; equationnumber < m; equationnumber++) {
			if (essential[i][equationnumber] == BoundaryConditions::dirichlet)
				(*defectGlobal)[i][equationnumber] = 0;
			}
	}

	virtual double injected(double& upperMass, double& oldUpperMass) {
		typedef typename G::Traits::template Codim<0>::Entity Entity;
		typedef typename G::ctype DT;
		typedef typename GV::template Codim<0>::Iterator Iterator;
		enum {dim = G::dimension};
		enum {dimworld = G::dimensionworld};

		const GV& gridview(this->grid_.leafView());
		double totalMass = 0;
		upperMass = 0;
		oldUpperMass = 0;
		// iterate through leaf grid an evaluate c0 at cell center
		Iterator eendit = gridview.template end<0>();
		for (Iterator it = gridview.template begin<0>(); it
				!= eendit; ++it) {
			// get geometry type
			Dune::GeometryType gt = it->geometry().type();

			// get entity
			const Entity& entity = *it;

			FVElementGeometry<G> fvGeom;
			fvGeom.update(entity);

			const typename Dune::LagrangeShapeFunctionSetContainer<DT,RT,dim>::value_type
					&sfs=Dune::LagrangeShapeFunctions<DT, RT, dim>::general(gt,
							1);
			int size = sfs.size();

			for (int i = 0; i < size; i++) {
				// get cell center in reference element
				const Dune::FieldVector<DT,dim>&local = sfs[i].position();

				// get global coordinate of cell center
				Dune::FieldVector<DT,dimworld> global = it->geometry().global(local);

				int globalId = vertexmapper.template map<dim>(entity,
						sfs[i].entity());

				double volume = fvGeom.subContVol[i].volume;

				double porosity = this->problem.soil().porosity(global, entity, local);

				double density = this->problem.materialLaw().nonwettingPhase.density();

				double mass = volume*porosity*density*((*(this->u))[globalId][1]);

				totalMass += mass;

				if (global[2] > 80.0) {
					upperMass += mass;
					oldUpperMass += volume*porosity*density*((*(this->uOldTimeStep))[globalId][1]);
				}
			}
		}

		return totalMass;
	}


    const G &grid() const
        { return grid_; }

protected:
	ProblemType& problem;
	const G& grid_;
	VertexMapper vertexmapper;
	int size;
	BlockVector<FieldVector<RT, 1> > pW;
	BlockVector<FieldVector<RT, 1> > pC;
	BlockVector<FieldVector<RT, 1> > satW;
	BlockVector<FieldVector<RT, 1> > satEx;
	BlockVector<FieldVector<RT, 1> > pEx;
	BlockVector<FieldVector<RT, 1> > satError;
};

}
#endif
