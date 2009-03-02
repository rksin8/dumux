#ifndef DUNE_MINCMODEL_HH
#define DUNE_MINCMODEL_HH

#include <dune/disc/shapefunctions/lagrangeshapefunctions.hh>
//#include <dune/disc/functions/p1function.hh>
#include "dumux/operators/p1operatorextended.hh"
#include "dumux/nonlinear/nonlinearmodel.hh"
#include "dumux/fvgeometry/fvelementgeometry.hh"

#include <boost/format.hpp>

namespace Dune {

/** \todo Please doc me! */

template<class G, class RT, class ProblemType, class LocalJacobian,
         class FunctionType, class OperatorAssembler> class MincModel :
        public NonlinearModel<G, RT, ProblemType, LocalJacobian, FunctionType, OperatorAssembler> {
public:
    typedef Dune::NonlinearModel<G, RT, ProblemType, LocalJacobian,
                                 FunctionType, OperatorAssembler> NonlinearModel;

    MincModel(const G& g, ProblemType& prob) :
        NonlinearModel(g, prob), uOldTimeStep(g) {
    }

    MincModel(const G& g, ProblemType& prob, int level) :
        NonlinearModel(g, prob, level), uOldTimeStep(g, level) {
    }

    virtual void initial() = 0;

    virtual void update(double& dt) = 0;

    virtual void solve() = 0;

    FunctionType uOldTimeStep;
};

/** \todo Please doc me! */

template<class G, class RT, class ProblemType, class LocalJac, int m> class LeafP1MincModel :
        public MincModel<G, RT, ProblemType, LocalJac,
                         Dune::LeafP1Function<G, RT, m>, LeafP1OperatorAssembler<G, RT, m> > {
public:
    // define the function type:
    typedef LeafP1Function<G, RT, m> FunctionType;

    // define the operator assembler type:
    typedef LeafP1OperatorAssembler<G, RT, m> OperatorAssembler;

    typedef Dune::MincModel<G, RT, ProblemType, LocalJac,
                            FunctionType, OperatorAssembler> MincModel;

    typedef LeafP1MincModel<G, RT, ProblemType, LocalJac, m> ThisType;

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

    LeafP1MincModel(const G& g, ProblemType& prob) :
        MincModel(g, prob), problem(prob), grid_(g), vertexmapper(g,
                                                                  g.leafIndexSet()), size((*(this->u)).size()), satExFracture(0), pExFracture(0), satErrorFracture(0) {
        for (int i = 0; i < m/2; ++i) {
            pWFracture[i].resize(size);
            pNFracture[i].resize(size);
            pCFracture[i].resize(size);
            satWFracture[i].resize(size);
            satNFracture[i].resize(size);
        }
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

                //                double porosity = this->problem.porosity(global, entity, local);
                double porosityFracture = this->problem.porosityFracture(global, entity, local);
                //                double porosityMatrix = this->problem.porosityMatrix(global, entity, local);

                double density = this->problem.materialLaw().nonwettingPhase.density();

                double mass = volume*porosityFracture*density*((*(this->u))[globalId][1]);

                totalMass += mass;

                if (global[2] > 80.0) {
                    upperMass += mass;
                    oldUpperMass += volume*porosityFracture*density*((*(this->uOldTimeStep))[globalId][1]);
                }
            }
        }

        return totalMass;
    }

    virtual void vtkout(const char* name, int k) {
        VTKWriter<typename G::LeafGridView> vtkwriter(this->grid_.leafView());
        char fname[128];
        sprintf(fname, "%s-%05d", name, k);
        double minSat = 1e100;
        double maxSat = -1e100;
        if (problem.exsolution) {
            satExFracture.resize(size);
            satErrorFracture.resize(size);
        }
        for (int j=0; j<m/2; j++){
            for (int i = 0; i < size; i++) {

                pWFracture[j][i] = (*(this->u))[i][j*2 + 0];
                satNFracture[j][i] = (*(this->u))[i][j*2 + 1];
                satWFracture[j][i] = 1 - satNFracture[j][i];

                double satNI = satNFracture[j][i];
                minSat = std::min(minSat, satNI);
                maxSat = std::max(maxSat, satNI);
                //pNFracture[i] = (*(this->u))[i][1];
                //pCFracture[i] = pNFracture[i] - pWFracture[i];
                //satWFracture[i] = this->problem.materialLaw().saturationW(pCFracture[i]);
                //satNFracture[i] = 1 - satWFracture[i];
                if (problem.exsolution) {
                    satExFracture[i]=problem.uExOutVertex(i, 1);
                    satErrorFracture[i]=problem.uExOutVertex(i, 2);
                }
            }
            vtkwriter.addVertexData(pWFracture[j], (boost::format("wetting phase pressure for continuum  %d")%j).str());
            vtkwriter.addVertexData(satWFracture[j], (boost::format("wetting phase saturation for continuum  %d")%j).str());
            vtkwriter.addVertexData(satNFracture[j], (boost::format("nonwetting phase saturation for continuum  %d")%j).str());
            //        vtkwriter.addVertexData(pWMatrix, "wetting phase pressure in fracture");
            if (problem.exsolution) {
                vtkwriter.addVertexData(satExFracture, "saturation, exact solution");
                vtkwriter.addVertexData(satErrorFracture, "saturation error");
            }
        }
        vtkwriter.write(fname, VTKOptions::ascii);
        std::cout << "nonwetting phase saturation: min = "<< minSat
                  << ", max = "<< maxSat << std::endl;
        if (minSat< -0.5 || maxSat > 1.5)DUNE_THROW(MathError, "Saturation exceeds range.");

    }

    const G& grid() const
    { return grid_; }


protected:
    ProblemType& problem;
    const G& grid_;
    VertexMapper vertexmapper;
    int size;
    //    BlockVector<FieldVector<RT, 1> > pW;
    BlockVector<FieldVector<RT, 1> > pWFracture[m/2];
    //    BlockVector<FieldVector<RT, 1> > pWMatrix[m/2];
    //    BlockVector<FieldVector<RT, 1> > pN;
    BlockVector<FieldVector<RT, 1> > pNFracture[m/2];
    //    BlockVector<FieldVector<RT, 1> > pNMatrix;
    //    BlockVector<FieldVector<RT, 1> > pC;
    BlockVector<FieldVector<RT, 1> > pCFracture[m/2];
    //    BlockVector<FieldVector<RT, 1> > pCMatrix;
    //    BlockVector<FieldVector<RT, 1> > satW;
    BlockVector<FieldVector<RT, 1> > satWFracture[m/2];
    //    BlockVector<FieldVector<RT, 1> > satWMatrix;
    //    BlockVector<FieldVector<RT, 1> > satN;
    BlockVector<FieldVector<RT, 1> > satNFracture[m/2];
    //    BlockVector<FieldVector<RT, 1> > satNMatrix;
    //    BlockVector<FieldVector<RT, 1> > satEx;
    BlockVector<FieldVector<RT, 1> > satExFracture;
    //    BlockVector<FieldVector<RT, 1> > satExMatrix;
    //    BlockVector<FieldVector<RT, 1> > pEx;
    BlockVector<FieldVector<RT, 1> > pExFracture;
    //    BlockVector<FieldVector<RT, 1> > pExMatrix;
    //    BlockVector<FieldVector<RT, 1> > satError;
    BlockVector<FieldVector<RT, 1> > satErrorFracture;
    //    BlockVector<FieldVector<RT, 1> > satErrorMatrix;
};

}
#endif
