// $Id$

#ifndef DUNE_DIFFUSIONVELOCITYPROBLEM_HH
#define DUNE_DIFFUSIONVELOCITYPROBLEM_HH

#include "dumux/diffusion/fv/fvdiffusion_pw.hh"

namespace Dune {

/** \todo Please doc me! */

template<class G, class RT, class VC> class DeprecatedFVDiffusionVelocity :
        public DeprecatedFVDiffusion<G, RT, VC> {

    typedef typename G::Traits::template Codim<0>::Entity Entity;
    typedef typename G::ctype ct;
    typedef typename G::LevelGridView GV;
    typedef typename GV::IndexSet IS;
    typedef typename GV::template Codim<0>::Iterator Iterator;
    typedef typename IntersectionIteratorGetter<G,LevelTag>::IntersectionIterator
    IntersectionIterator;
    typedef typename G::template Codim<0>::EntityPointer EntityPointer;

    enum {dim = G::dimension};
    enum {dimworld = G::dimensionworld};

public:
    DeprecatedFVDiffusionVelocity(G& g, DeprecatedDiffusionProblem<G, RT, VC>& prob, int lev = -1)
        : DeprecatedFVDiffusion<G,RT,VC>(g, prob, lev)
    {    }


    void calcTotalVelocity(const RT t=0) const {
        // find out whether gravity effects are relevant
        bool hasGravity = false;
        const FieldVector<ct,dim>& gravity(this->diffproblem.gravity());
        for (int k = 0; k < dim; k++)
            if (gravity[k] != 0)
                hasGravity = true;

        Iterator eendit = this->gridview.template end<0>();
        for (Iterator it = this->gridview.template begin<0>(); it
                 != eendit; ++it) {
            // cell geometry type
            GeometryType gt = it->geometry().type();

            // cell center in reference element
            const FieldVector<ct,dim>
                &local = ReferenceElements<ct,dim>::general(gt).position(0, 0);

            // cell center in global coordinates
            const FieldVector<ct,dimworld>global = it->geometry().global(local);

            // cell index
            int indexi = this->elementmapper.map(*it);

            // get pressure and permeability in element
            double pressi = this->diffproblem.variables.pressure[indexi];

            // get absolute permeability
            FieldMatrix<ct,dim,dim> Ki(this->diffproblem.K(global, *it, local));

            //compute total mobility
            double lambdaI; //, fractionalWI;
            double sati = this->diffproblem.variables.saturation[indexi];

            lambdaI = this->diffproblem.materialLaw.mobW(sati);

            double faceVol[2*dim];

            // run through all intersections with neighbors and boundary
            IntersectionIterator endit = IntersectionIteratorGetter<G, LevelTag>::end(*it);
            for (IntersectionIterator is = IntersectionIteratorGetter<G,
                     LevelTag>::begin(*it); is!=endit; ++is) {
                // get geometry type of face
                GeometryType gtf = is->intersectionSelfLocal().type();

                //Geometry dg = is.intersectionSelfLocal();
                // local number of facet
                int numberInSelf = is->numberInSelf();

                switch (G::dimension) {
                case 1:
                    faceVol[numberInSelf] = 1;
                default:
                    faceVol[numberInSelf] = is->intersectionGlobal().volume();
                }

                // center in face's reference element
                const FieldVector<ct,dim-1>&
                    facelocal = ReferenceElements<ct,dim-1>::general(gtf).position(0,0);

                // center of face inside volume reference element
                const FieldVector<ct,dim>&
                    facelocalDim = ReferenceElements<ct,dim>::general(gtf).position(numberInSelf,1);

                // get normal vector
                FieldVector<ct,dimworld> unitOuterNormal
                    = is->unitOuterNormal(facelocal);

                // center of face in global coordinates
                FieldVector<ct,dimworld>
                    faceglobal = is->intersectionGlobal().global(facelocal);

                // handle interior face
                if (is->neighbor())
                    {
                        // access neighbor
                        EntityPointer outside = is->outside();
                        int indexj = this->elementmapper.map(*outside);

                        // get neighbor pressure and permeability
                        double pressj = this->diffproblem.variables.pressure[indexj];

                        // compute factor in neighbor
                        GeometryType nbgt = outside->geometry().type();
                        const FieldVector<ct,dim>&
                            nblocal = ReferenceElements<ct,dim>::general(nbgt).position(0,0);

                        // neighbor cell center in global coordinates
                        FieldVector<ct,dimworld>
                            nbglobal = outside->geometry().global(nblocal);

                        // distance vector between barycenters
                        FieldVector<ct,dimworld> distVec = global - nbglobal;

                        // compute distance between cell centers
                        double dist = distVec.two_norm();

                        // get absolute permeability
                        FieldMatrix<ct,dim,dim> Kj(this->diffproblem.K(nbglobal, *outside, nblocal));

                        // compute vectorized permeabilities
                        FieldVector<ct,dim> Kni(0);
                        FieldVector<ct,dim> Knj(0);
                        Ki.umv(unitOuterNormal, Kni);
                        Kj.umv(unitOuterNormal, Knj);
                        // compute permeability normal to intersection and take harmonic mean
                        double K_n_i = Kni * unitOuterNormal;
                        double K_n_j = Knj * unitOuterNormal;
                        double Kn = 2 * K_n_i * K_n_j / (K_n_i + K_n_j);
                        // compute permeability tangential to intersection and take arithmetic mean
                        FieldVector<ct,dim> uON = unitOuterNormal;
                        FieldVector<ct,dim> K_t_i = Kni - (uON *= K_n_i);
                        uON = unitOuterNormal;
                        FieldVector<ct,dim> K_t_j = Knj - (uON *= K_n_j);
                        FieldVector<ct,dim> Kt = (K_t_i += K_t_j);
                        Kt *= 0.5;
                        // Build vectorized averaged permeability
                        uON = unitOuterNormal;
                        FieldVector<ct,dim> K = (Kt += (uON *=Kn));

                        //compute total mobility
                        double lambdaJ; //, fractionalWJ;
                        double satj = this->diffproblem.variables.saturation[indexj];
                        double satI = sati;
                        double satJ = satj;

                        for (int k = 0; k < dim; k++)
                            if (fabs(distVec[k]) > 0.5*dist)
                                {
                                    satI -= fabs(distVec[k])/distVec[k]*0.5*dist*this->diffproblem.variables.slope[indexi][k];
                                    satJ += fabs(distVec[k])/distVec[k]*0.5*dist*this->diffproblem.variables.slope[indexj][k];
                                }

                        lambdaI = this->diffproblem.materialLaw.mobW(satI);
                        lambdaJ = this->diffproblem.materialLaw.mobW(satJ);

                        // compute averaged total mobility
                        // CAREFUL: Harmonic weightig can generate zero matrix entries,
                        // use arithmetic weighting instead:
                        double lambda = 1;

                        FieldVector<ct,dimworld> vTotal(K);
                        double pressgrad = 0;
                        //                    double pcgrad = 0;

                        pressgrad = (pressi - pressj)/dist;

                        ////                    std::cout<<"(pressGrad-pcgrad) = "<<pressgrad<<std::endl;
                        //                    if ((pressgrad-pcgrad) >= 0)
                        //                        lambda=lambdaI;
                        //                    else
                        //                        lambda=lambdaJ;

                        lambda = 0.5*(lambdaI+lambdaJ);

                        vTotal *= (lambda*(pressgrad));

                        if (hasGravity) {
                            Ki += Kj;
                            Ki *= 0.5;
                            FieldVector<ct,dimworld> gEffect(0);
                            Ki.umv(gravity, gEffect);
                            gEffect *= lambda*this->diffproblem.materialLaw.wettingPhase.density();
                            vTotal += gEffect;
                        }
                        this->diffproblem.variables.velocity[indexi][numberInSelf] = vTotal;
                    }
                // boundary face
                else
                    {
                        //get boundary condition for boundary face center
                        BoundaryConditions::Flags bctype = this->diffproblem.bctype(faceglobal, *it, facelocalDim);
                        if (bctype == BoundaryConditions::dirichlet) {
                            // distance vector between barycenters
                            FieldVector<ct,dimworld> distVec = global - faceglobal;

                            double dist = distVec.two_norm();
                            distVec /= dist;

                            // compute directed permeability vector Ki.n
                            FieldVector<ct,dim> Kni(0);
                            Ki.umv(distVec, Kni);

                            // compute averaged total mobility
                            double lambda = 1.;
                            //                        double satI = sati;

                            lambda = this->diffproblem.materialLaw.mobW(this->diffproblem.dirichletSat(faceglobal, *it, facelocalDim));

                            double g = this->diffproblem.dirichletPress(faceglobal, *it, facelocalDim);

                            FieldVector<ct,dim> vTotal(Kni);
                            double pressgrad = 0;

                            pressgrad = (pressi-g)/dist;

                            vTotal *= -(lambda*pressgrad);

                            if (hasGravity) {
                                FieldVector<ct,dimworld> gEffect(0);
                                Ki.umv(gravity, gEffect);
                                gEffect *= lambda*this->diffproblem.materialLaw.wettingPhase.density();
                                vTotal -= gEffect;
                            }
                            this->diffproblem.variables.velocity[indexi][numberInSelf] = vTotal;
                        }
                        else
                            {
                                double J = this->diffproblem.neumannPress(faceglobal, *it, facelocalDim);
                                FieldVector<ct,dimworld> unitOuterNormal
                                    = is->unitOuterNormal(facelocal);
                                this->diffproblem.variables.velocity[indexi][numberInSelf] = unitOuterNormal;
                                this->diffproblem.variables.velocity[indexi][numberInSelf] *= J;
                            }

                    }
            }
            // end all intersections
            //            std::cout<<"velocity = "<< this->diffproblem.variables.velocity <<std::endl;
            if (dim == 1&& this->diffproblem.capillarity != true) {
                double sum = (fabs(this->diffproblem.variables.velocity[indexi][0][0]*faceVol[0])
                              + fabs(this->diffproblem.variables.velocity[indexi][1][0]));
                double diff = fabs(this->diffproblem.variables.velocity[indexi][0][0]*faceVol[0]
                                   - this->diffproblem.variables.velocity[indexi][1][0]*faceVol[1])/sum;
                if (diff > 1e-6&& sum > 1e-9) {
                    std::cout << "NOT conservative!!! diff = "<< diff
                              << ", indexi = "<< indexi << std::endl;
                    std::cout << this->diffproblem.variables.velocity[indexi][0][0]*faceVol[0]<< ", "
                              << this->diffproblem.variables.velocity[indexi][1][0]*faceVol[1]<< std::endl;
                }
            }
            if (dim == 2&& this->diffproblem.capillarity != true) {
                double sum = (fabs(this->diffproblem.variables.velocity[indexi][0][0]*faceVol[0])
                              + fabs(this->diffproblem.variables.velocity[indexi][1][0]*faceVol[1])
                              + fabs(this->diffproblem.variables.velocity[indexi][2][1]*faceVol[2])
                              + fabs(this->diffproblem.variables.velocity[indexi][3][1]*faceVol[3]));
                double diff = fabs(this->diffproblem.variables.velocity[indexi][0][0]*faceVol[0]
                                   - this->diffproblem.variables.velocity[indexi][1][0]*faceVol[1]
                                   + this->diffproblem.variables.velocity[indexi][2][1]*faceVol[2]
                                   - this->diffproblem.variables.velocity[indexi][3][1]*faceVol[3])/sum;
                if (diff > 1e-6&& sum > 1e-9) {
                    std::cout << "NOT conservative!!! diff = "<< diff
                              << ", indexi = "<< indexi << std::endl;
                    std::cout << this->diffproblem.variables.velocity[indexi][0][0]*faceVol[0]<< ", "
                              << this->diffproblem.variables.velocity[indexi][1][0]*faceVol[1]<< ", "
                              << this->diffproblem.variables.velocity[indexi][2][1]*faceVol[2]<< ", "
                              << this->diffproblem.variables.velocity[indexi][3][1]*faceVol[3]<< std::endl;
                }
            }
        } // end grid traversal
        return;
    }
};
}
#endif
