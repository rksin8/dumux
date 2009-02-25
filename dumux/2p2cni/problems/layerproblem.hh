// $Id$

#ifndef DUNE_LAYERPROBLEM_HH
#define DUNE_LAYERPROBLEM_HH

#include<iostream>
#include<iomanip>

#include<dune/common/fvector.hh>
#include<dune/common/fmatrix.hh>
#include<dune/common/exceptions.hh>
#include<dune/grid/common/grid.hh>
#include<dune/grid/common/referenceelements.hh>
#include<dune/disc/operators/boundaryconditions.hh>
#include<dumux/material/twophaserelations_deprecated.hh>
#include<dumux/material/linearlaw_deprecated.hh>
#include<dumux/2p2c/2p2cproblem.hh>

/**
 * @file
 * @brief  Base class for defining an instance of the TwoPhase problem
 * @author Bernd Flemisch
 */

namespace Dune
{
  //! base class that defines the parameters of a diffusion equation
  /*! An interface for defining parameters for the stationary diffusion equation
   * \f$ - \text{div}\, (\lambda K \text{grad}\, p ) = q, \f$,
   * \f$p = g\f$ on \f$\Gamma_1\f$, and \f$\lambda K \text{grad}\, p = J\f$
   * on \f$\Gamma_2\f$. Here,
   * \f$p\f$ denotes the pressure, \f$K\f$ the absolute permeability,
   * and \f$\lambda\f$ the total mobility, possibly depending on the
   * saturation.
   *
   *    Template parameters are:
   *
   *    - Grid  a DUNE grid type
   *    - RT    type used for return values
   */
  template<class G, class RT>
  class LayerProblem : public TwoPTwoCProblem<G, RT> {
    typedef typename G::ctype DT;
    enum {dim=G::dimension, m=2};
    typedef typename G::Traits::template Codim<0>::Entity Entity;
    typedef typename IntersectionIteratorGetter<G,LeafTag>::IntersectionIterator IntersectionIterator;

  public:
    enum {pWIdx = 0, satNIdx = 1};
    enum {swrIdx = 0, snrIdx = 1, alphaIdx = 2, nIdx = 3};
    enum {gasPhase = 0, waterPhase = 1, bothPhases = 2};

    // permeabilities
    virtual const FieldMatrix<DT,dim,dim>& K (const FieldVector<DT,dim>& x)
    //, const Entity& e, const FieldVector<DT,dim>& xi)
    {
        if (x[0] >= innerLowerLeft_[0] && x[0] <= innerUpperRight_[0]
            && x[1] >= innerLowerLeft_[1] && x[1] <= innerUpperRight_[1])
            return innerK_;
        else
            return outerK_;
    }

    // sources and sinks
    virtual FieldVector<RT,m> q (const FieldVector<DT,dim>& x, const Entity& e,
                    const FieldVector<DT,dim>& xi) const
    {
        FieldVector<RT,m> values(0);

        return values;
    }

/////////////////////////////
// TYPE of the boundaries
/////////////////////////////
    virtual FieldVector<BoundaryConditions::Flags, m> bctype (const FieldVector<DT,dim>& x, const Entity& e,
                    const IntersectionIterator& intersectionIt,
                       const FieldVector<DT,dim>& xi) const
    {
        FieldVector<BoundaryConditions::Flags, m> values(BoundaryConditions::neumann);

        if (x[0] < outerLowerLeft_[0] + eps_)
            values = BoundaryConditions::dirichlet;
//        if (x[1] < outerLowerLeft_[1] + eps_)
//            values = BoundaryConditions::dirichlet;

        return values;
    }

/////////////////////////////
// DIRICHLET boundaries
/////////////////////////////
    virtual FieldVector<RT,m> g (const FieldVector<DT,dim>& x, const Entity& e,
                const IntersectionIterator& intersectionIt,
                  const FieldVector<DT,dim>& xi) const
    {
        FieldVector<RT,m> values(0);

        values[pWIdx] = -densityW_*gravity_[1]*(depthBOR_ - x[1]);
//        if (x[1] >= innerLowerLeft_[1] && x[1] <= innerUpperRight_[1])
        if (x[0] >= innerLowerLeft_[0])
            values[satNIdx] = 0.2;
        else
            values[satNIdx] = 1e-6;

        return values;
    }

/////////////////////////////
// NEUMANN boundaries
/////////////////////////////
    virtual FieldVector<RT,m> J (const FieldVector<DT,dim>& x, const Entity& e,
                const IntersectionIterator& intersectionIt,
                  const FieldVector<DT,dim>& xi) const
    {
        FieldVector<RT,m> values(0);

        //RT lambda = (x[1])/height_;
        if (x[1] < 0.25 && x[1] > outerLowerLeft_[0] + eps_)
            values[satNIdx] = 0.0;//-5e-7;

        return values;
    }

/////////////////////////////
// INITIAL values
/////////////////////////////
    virtual FieldVector<RT,m> initial (const FieldVector<DT,dim>& x, const Entity& e,
                  const FieldVector<DT,dim>& xi) const
    {

        FieldVector<RT,m> values;

        values[pWIdx] = -densityW_*gravity_[1]*(depthBOR_ - x[1]);

        if (x[1] >= innerLowerLeft_[1] && x[1] <= innerUpperRight_[1]
              && x[0] >= innerLowerLeft_[0])
            values[satNIdx] = 0.2;
        else
            values[satNIdx] = 1e-6;

        return values;
    }


    int initialPhaseState (const FieldVector<DT,dim>& x, const Entity& e,
                  const FieldVector<DT,dim>& xi) const
    {

        enum {gasPhase = 0, waterPhase = 1, bothPhases = 2}; // Phase states
        int state;

        if (x[1] >= innerLowerLeft_[1] && x[1] <= innerUpperRight_[1]
              && x[0] >= innerLowerLeft_[0])
            state = bothPhases;
        else
            state = waterPhase;

        return state;
    }

///////////////////////////////

    double porosity (const FieldVector<DT,dim>& x, const Entity& e,
              const FieldVector<DT,dim>& xi) const
    {
        if (x[0] > innerLowerLeft_[0] && x[0] < innerUpperRight_[0]
            && x[1] > innerLowerLeft_[1] && x[1] < innerUpperRight_[1])
            return innerPorosity_;
        else
            return outerPorosity_;
    }

    virtual FieldVector<RT,dim> gravity () const
    {
        return gravity_;
    }

    double depthBOR () const
    {
        return depthBOR_;
    }

    virtual FieldVector<RT,4> materialLawParameters (const FieldVector<DT,dim>& x, const Entity& e,
              const FieldVector<DT,dim>& xi) const
    {
        FieldVector<RT,4> values;

        if (x[0] > innerLowerLeft_[0] && x[0] < innerUpperRight_[0]
            && x[1] > innerLowerLeft_[1] && x[1] < innerUpperRight_[1]) {
            values[swrIdx] = innerSwr_;
            values[snrIdx] = innerSnr_;
            values[alphaIdx] = innerAlpha_;
            values[nIdx] = innerN_;
        }
        else {
            values[swrIdx] = outerSwr_;
            values[snrIdx] = outerSnr_;
            values[alphaIdx] = outerAlpha_;
            values[nIdx] = outerN_;
        }

        return values;
    }

    LayerProblem(TwoPhaseRelations& law = *(new DeprecatedLinearLaw), MultiComp& multicomp = *(new CWaterAir),
            const FieldVector<DT,dim> outerLowerLeft = 0., const FieldVector<DT,dim> outerUpperRight = 0.,
            const FieldVector<DT,dim> innerLowerLeft = 0., const FieldVector<DT,dim> innerUpperRight = 0.,
            const RT depthBOR = 0., RT outerK = 1.2e-12, RT innerK = 1.2e-12,
            RT outerSwr = 0.05, RT outerSnr = 0.1, RT innerSwr = 0.05, RT innerSnr = 0.1,
            RT outerPorosity = 0.4, RT innerPorosity = 0.4,
            RT outerAlpha = 0.0037, RT innerAlpha = 0.0037,  //0.00045
            RT outerN = 4.7, RT innerN = 4.7)    //7.3
    : TwoPTwoCProblem<G, RT>(law, multicomp),
      outerLowerLeft_(outerLowerLeft), outerUpperRight_(outerUpperRight),
      innerLowerLeft_(innerLowerLeft), innerUpperRight_(innerUpperRight),
      depthBOR_(depthBOR), eps_(1e-8*outerUpperRight[0]),
      densityW_(law.wettingPhase.density()), densityN_(law.nonwettingPhase.density()),
      outerSwr_(outerSwr), outerSnr_(outerSnr), innerSwr_(innerSwr), innerSnr_(innerSnr),
      outerPorosity_(outerPorosity), innerPorosity_(innerPorosity),
      outerAlpha_(outerAlpha), innerAlpha_(innerAlpha),
      outerN_(outerN), innerN_(innerN)
    {
        outerK_[0][0] = outerK_[1][1] = outerK;
        outerK_[0][1] = outerK_[1][0] = 0;

        innerK_[0][0] = innerK_[1][1] = innerK;
        innerK_[0][1] = innerK_[1][0] = 0;

        height_ = outerUpperRight[1] - outerLowerLeft[1];
        width_ = outerUpperRight[0] - outerLowerLeft[0];

        gravity_[0] = 0;
        gravity_[1] = -9.81;
    }

    private:
        FieldMatrix<DT,dim,dim> outerK_;
        FieldMatrix<DT,dim,dim> innerK_;
        FieldVector<DT,dim> outerLowerLeft_;
        FieldVector<DT,dim> outerUpperRight_;
        FieldVector<DT,dim> innerLowerLeft_;
        FieldVector<DT,dim> innerUpperRight_;
        DT width_, height_;
        DT depthBOR_, eps_;
        RT densityW_, densityN_;
        FieldVector<DT,dim> gravity_;
        RT outerSwr_, outerSnr_, innerSwr_, innerSnr_;
        RT outerPorosity_, innerPorosity_;
        RT outerAlpha_, innerAlpha_;
        RT outerN_, innerN_;
  };

}
#endif
