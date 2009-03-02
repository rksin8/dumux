// $Id$

#ifndef DUNE_TWOPHASERELATIONS_HH
#define DUNE_TWOPHASERELATIONS_HH

#include <dumux/material/property_baseclasses.hh>
#include <dumux/material/relperm_pc_law.hh>

/**
 * \author Jochen Fritz
 * \ingroup material
 * \defgroup 2pRel Relative Permeabilities and material properties
 */


namespace Dune
{
/*!\ingroup 2pRel
 * \brief Provides all relative permeability and capillary pressure relationships
 *
 *  Depending on the saturation \f$S_\text{w}\f$ and the viscosities \f$\mu_\text{w}\f$,\f$\mu_\text{n}\f$,
 *  the phase mobilities \f$\lambda_\text{w}\f$, \f$\lambda_\text{n}\f$,
 *  the total mobility \f$\lambda\f$, the capillary pressure \f$p_\text{c}\f$, and its derivative
 *  \f$\text{d}p_\text{c}/\text{d}S_\text{e}\f$ can be calculated.
 *  Linear, Brooks-Corey, Van Genuchten or auxiliary Laws are chosen automaticaly due to flags which are
 *  specified in the member soil (class Matrix2p).
 *  Linear, Brooks-Corey and Van Genuchten Laws as derivated classes of RelPerm_pc are already members
 *  of the class. Auxiliary relative permeability / capillary pressure saturation relationships can be
 *  added as constructor arguments.
 */
template<class G, class RT>
class TwoPhaseRelations
{
public:

    typedef typename G::Traits::template Codim<0>::Entity Entity;
    typedef typename G::ctype DT;
    enum { n=G::dimension};

    /** \brief constructor
     *  \param s soil properties
     *  \param wP wetting phase properties
     *  \param nP nonwetting phase properties
     *  \param a1 auxiliary relative permeability capillary pressure saturation relationship
     *  \param a2 auxiliary relative permeability capillary pressure saturation relationship
     *  \param a3 auxiliary relative permeability capillary pressure saturation relationship
     */
    TwoPhaseRelations(Matrix2p<G, RT>& s, Fluid& wP , Fluid& nwP
                      /*,RelPerm_pc<G>& a1 = (*new LinearLaw<G>(s, false)),
                        RelPerm_pc<G>& a2 = (*new LinearLaw<G>(s, false)),
                        RelPerm_pc<G>& a3 = (*new LinearLaw<G>(s, false))*/)
        : wettingPhase(wP), nonwettingPhase(nwP), soil(s),
          brookscorey(s, false), vangenuchten(s, false), linearlaw(s, false),
          auxiliary1(*new LinearLaw<G>(s, false)), auxiliary2(*new LinearLaw<G>(s, false)),
          auxiliary3(*new LinearLaw<G>(s, false))
    {     }

    virtual ~TwoPhaseRelations()
    {
        delete &(auxiliary1);
        delete &(auxiliary2);
        delete &(auxiliary3);
    }

    Fluid& wettingPhase; //!< contains the properties of the wetting phase
    Fluid& nonwettingPhase; //!< contains the properties of the nonwetting phase
    Matrix2p<G, RT>& soil;

protected:
    const BrooksCoreyLaw<G> brookscorey;
    const VanGenuchtenLaw<G> vangenuchten;
    const LinearLaw<G> linearlaw;
    const RelPerm_pc<G>& auxiliary1;
    const RelPerm_pc<G>& auxiliary2;
    const RelPerm_pc<G>& auxiliary3;

public:
    /*! \brief Implements the wetting phase mobility/saturation relation.
     *  \param saturationW the saturation of the wetting phase
     *  \param T temperature
     *  \param p pressure
     *  \param Xa mass fraction of the non-wetting component in the wetting phase
     *  \return the mobility of the wetting phase
     */
    double mobW (double saturationW,
                 const FieldVector<DT,n>& x,
                 const Entity& e,
                 const FieldVector<DT,n>& xi,
                 double T = 283.15,
                 double p = 1e5,
                 double Xa = 0.0) const
    {
        double viscosityW = wettingPhase.viscosity ( T, p, Xa);
        return krw(saturationW, x, e, xi, T) / viscosityW;
    }

    /*! \brief Implements the nonwetting phase mobility/saturation relation.
     *  \param saturationN the saturation of the nonwetting phase
     *  \param T temperature
     *  \param p pressure
     *  \param Xw mass fraction of the wetting component in the non-wetting phase
     *  \return the mobility of the nonwetting phase
     */
    double mobN (double saturationN,
                 const FieldVector<DT,n>& x,
                 const Entity& e,
                 const FieldVector<DT,n>& xi,
                 double T=283.15,
                 double p=1e5,
                 double Xw = 0) const
    {
        double viscosityN = nonwettingPhase.viscosity ( T, p, Xw);
        return krn(saturationN, x, e, xi, T) / viscosityN;
    }

    /*! \brief Implements the total mobility/saturation relation.
     *  Assumption: Phases do not mix and mixing effects do not influence viscosity respectively.
     *  \param saturationW the saturation of the wetting phase
     *  \param T temperature
     *  \param p pressure
     *  \return the total mobility
     */
    double mobTotal (double saturationW, const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi, double T=283.15, double p=1e5) const
    {
        return mob(saturationW, x, e, xi, T, p)[2];
    }

    /*! \brief mobility/saturation relation for both phases AND total mobility
     *  If the mobilities of both phases are needed, calling this function is more efficient than calling
     *  mobW and mobN!
     *  Assumption: Phases do not mix and mixing effects do not influence viscosity respectively.
     *  \param saturationW the saturation of the wetting phase
     *  \param T temperature
     *  \param p pressure
     *  \return mobilities in the order: mobility of wetting phase, mobility of nonwetting phase, total mobility
     */
    std::vector<double> mob (double saturationW, const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi, double T=283.15, double p=1e5) const
    {
        // phase viscosities
        double viscosityW = wettingPhase.viscosity ( T, p, 0.);
        double viscosityN = nonwettingPhase.viscosity ( T, p, 0.);

        // get relative permeabilities and divide by respective viscosities
        std::vector<double> lambda = kr(saturationW,x, e, xi, T);
        lambda[0] /= viscosityW;
        lambda[1] /= viscosityN;

        // add total mobility to the end of the vector
        lambda.resize(3);
        lambda[2] = lambda[0] + lambda[1];
        return lambda;
    }

    /*! \brief Implements the wetting phase fractional flow function
     *  If the fractional flow functions of both phases are needed,
     *  calling this function is more efficient than calling fractionalW and fractionalN!
     *  Assumption: Phases do not mix and mixing effects do not influence viscosity respectively.
     *  \param saturationW the saturation of the wetting phase
     *  \param T temperature
     *  \param p pressure
     *  \return the fractional flow functions in the order: wetting phase, nonwetting phase
     */
    std::vector<double> fractional (double saturationW, const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi, double T=283.15, double p=1e5) const
    {
        // phase viscosities
        double viscosityW = wettingPhase.viscosity ( T, p, 0.);
        double viscosityN = nonwettingPhase.viscosity ( T, p, 0.);

        // get relative permeabilities and divide by respective viscosities
        std::vector<double> f = kr(saturationW,x, e, xi, T);
        f[0] /= viscosityW;
        f[1] /= viscosityN;

        double lambda_tot = f[0] + f[1];
        f[0] /= lambda_tot;
        f[1] /= lambda_tot;
        return f;
    }

    /*! \brief Implements the wetting phase fractional flow function
     *  Assumption: Phases do not mix and mixing effects do not influence viscosity respectively.
     *  \param saturationW the saturation of the wetting phase
     *  \param T temperature
     *  \param p pressure
     *  \return the fractional flow of the wetting phase
     */
    double fractionalW (double saturationW, const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi, double T=283.15, double p=1e5) const
    {
        return fractional(saturationW,x, e, xi, T)[0];
    }

    /*! \brief Implements the nonwetting phase fractional flow function
     *  Assumption: Phases do not mix and mixing effects do not influence viscosity respectively.
     *  \param saturationN the saturation of the nonwetting phase
     *  \param T temperature
     *  \param p pressure
     *  \return the fractional flow of the nonwetting phase
     */
    double fractionalN (double saturationN, const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi, double T=283.15, double p=1e5) const
    {
        return fractional(saturationW,x, e, xi, T)[1];
    }

    /*! \brief the capillary pressure - saturation relation
     *
     *  \param saturationW the saturation of the wetting phase
     *  \return the capillary pressure \f$ p_\text{c} (S_\text{w})\f$.
     */
    virtual double pC (double saturationW, const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi, double T = 283.15) const
    {
        const std::vector<double>& param = soil.paramRelPerm(x, e, xi);

        switch (soil.relPermFlag(x, e, xi))
            {
            case 0:
                return linearlaw.pC(saturationW, x, e, xi, param, T);
            case 1:
                return brookscorey.pC(saturationW, x, e, xi, param, T);
            case 2:
                return vangenuchten.pC(saturationW, x, e, xi, param, T);
            case 3:
                return auxiliary1.pC(saturationW, x, e, xi, param, T);
            case 4:
                return auxiliary2.pC(saturationW, x, e, xi, param, T);
            case 5:
                return auxiliary3.pC(saturationW, x, e, xi, param, T);
            default:
                DUNE_THROW(NotImplemented, "Matrix2p::modelFlag " << soil.relPermFlag(x, e, xi) << " for TwoPhaseRelations::pC");
            }
    }

    /*! \brief class for a second capillary pressure - saturation relation
     *
     *  \param saturationW the saturation of the wetting phase
     *  \param param standard vector containing the parameters for the material law
     *  \return the capillary pressure \f$ p_\text{c} (S_\text{w})\f$.
     */
    virtual double pC (double saturationW, const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi,
                       const std::vector<double>& param, double T = 283.15) const
    {
        // ATTENTION: relPermFlag2 is used from the soil class!
        // This function is mainly implemented, for the definition of a second
        // Pc-Sw relation for interfacial area models
        switch (soil.relPermFlag(x, e, xi))
            {
            case 0:
                return linearlaw.pC(saturationW, x, e, xi, param, T);
            case 1:
                return brookscorey.pC(saturationW, x, e, xi, param, T);
            case 2:
                return vangenuchten.pC(saturationW, x, e, xi, param, T);
            case 3:
                return auxiliary1.pC(saturationW, x, e, xi, param, T);
            case 4:
                return auxiliary2.pC(saturationW, x, e, xi, param, T);
            case 5:
                return auxiliary3.pC(saturationW, x, e, xi, param, T);
            default:
                DUNE_THROW(NotImplemented, "Matrix2p::modelFlag " << soil.relPermFlag(x, e, xi) << " for TwoPhaseRelations::pC");
            }
    }

    /*! \brief the derivative of capillary pressure w.r.t. the saturation
     *
     *  \param saturationW the saturation of the wetting phase
     *  \param T temperature
     *  \param p pressure
     *  \return the derivative \f$\text{d}p_\text{c}/\text{d}S_\text{e}\f$
     */
    virtual double dPdS (double saturationW, const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi, double T=283.15, double p=1e5) const
    {
        const std::vector<double>& param = soil.paramRelPerm(x, e, xi);

        switch (soil.relPermFlag(x, e, xi))
            {
            case 0:
                return linearlaw.dPdS(saturationW, x, e, xi, param, T);
            case 1:
                return brookscorey.dPdS(saturationW, x, e, xi, param, T);
            case 2:
                return vangenuchten.dPdS(saturationW, x, e, xi, param, T);
            case 3:
                return auxiliary1.dPdS(saturationW, x, e, xi, param, T);
            case 4:
                return auxiliary2.dPdS(saturationW, x, e, xi, param, T);
            case 5:
                return auxiliary3.dPdS(saturationW, x, e, xi, param, T);
            default:
                DUNE_THROW(NotImplemented, "Matrix2p::modelFlag " << soil.relPermFlag(x, e, xi) << " for TwoPhaseRelations::dPdS");
            }
    }

    /*! \brief second class for calculation of the derivative of capillary pressure w.r.t. the saturation
     *
     *  \param saturationW the saturation of the wetting phase
     *  \param param standard vector containing the parameters for the material law
     *  \param T temperature
     *  \param p pressure
     *  \return the derivative \f$\text{d}p_\text{c}/\text{d}S_\text{e}\f$
     */
    virtual double dPdS (double saturationW, const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi,
                         const std::vector<double>& param, double T=283.15, double p=1e5) const
    {
        switch (soil.relPermFlag(x, e, xi))
            {
            case 0:
                return linearlaw.dPdS(saturationW, x, e, xi, param, T);
            case 1:
                return brookscorey.dPdS(saturationW, x, e, xi, param, T);
            case 2:
                return vangenuchten.dPdS(saturationW, x, e, xi, param, T);
            case 3:
                return auxiliary1.dPdS(saturationW, x, e, xi, param, T);
            case 4:
                return auxiliary2.dPdS(saturationW, x, e, xi, param, T);
            case 5:
                return auxiliary3.dPdS(saturationW, x, e, xi, param, T);
            default:
                DUNE_THROW(NotImplemented, "Matrix2p::modelFlag " << soil.relPermFlag(x, e, xi) << " for TwoPhaseRelations::dPdS");
            }
    }

    /*! \brief the wetting phase saturation w.r.t. the capillary pressure
     *
     *  \param pC the capillary pressure
     *  \param T temperature
     *  \return the wetting phase saturation
     */
    virtual double saturationW (double pC, const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi, double T=283.15) const
    {
        switch (soil.relPermFlag(x, e, xi))
            {
            case 0:
                return linearlaw.saturationW(pC, x, e, xi, T);
            case 1:
                return brookscorey.saturationW(pC, x, e, xi, T);
            case 2:
                return vangenuchten.saturationW(pC, x, e, xi, T);
            case 3:
                return auxiliary1.saturationW(pC, x, e, xi, T);
            case 4:
                return auxiliary2.saturationW(pC, x, e, xi, T);
            case 5:
                return auxiliary3.saturationW(pC, x, e, xi, T);
            default:
                DUNE_THROW(NotImplemented, "Matrix2p::modelFlag " << soil.relPermFlag(x, e, xi) << " for TwoPhaseRelations::saturationW");
            }
    }

    /*! \brief the derivative of the saturation w.r.t. the capillary pressure
     *
     *  \param pC the capillary pressure
     *  \param T temperature
     *  \return the derivative \f$\text{d}S_w/\text{d}p_\text{c}\f$
     */
    virtual double dSdP (double pC, const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi, double T=283.15) const
    {
        switch (soil.relPermFlag(x, e, xi))
            {
            case 0:
                return linearlaw.dSdP(pC, x, e, xi, T);
            case 1:
                return brookscorey.dSdP(pC, x, e, xi, T);
            case 2:
                return vangenuchten.dSdP(pC, x, e, xi, T);
            case 3:
                return auxiliary1.dSdP(pC, x, e, xi, T);
            case 4:
                return auxiliary2.dSdP(pC, x, e, xi, T);
            case 5:
                return auxiliary3.dSdP(pC, x, e, xi, T);
            default:
                DUNE_THROW(NotImplemented, "Matrix2p::modelFlag " << soil.relPermFlag(x, e, xi) << " for TwoPhaseRelations::dSdP");
            }
    }

    /*! \brief the specific interfacial area function
     *
     *  \param pC the capillary pressure
     *  \param saturationW the saturation of the wetting phase
     *  \param awnparam the parameter of the biquadratic awn surface curve
     *  \return the derivative \f$\text{d}S_w/\text{d}p_\text{c}\f$
     */
    virtual double awn (double saturationW, double pC, const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi) const
    {
        const std::vector<double> awnparam = soil.awnParam(x, e, xi);
        double value;
        double a0,a1,a2,a3,a4,a5;
        a0=awnparam[0];
        a1=awnparam[1];
        a2=awnparam[2];
        a3=awnparam[3];
        a4=awnparam[4];
        a5=awnparam[5];

        value = awnparam[0]+awnparam[1]*saturationW + awnparam[2]*pC + awnparam[3]*pC*saturationW + awnparam[4]*saturationW*saturationW + awnparam[5]*pC*pC;

        return value;
    }

    /*! \brief the derivative of specific interfacial area function w.r.t. capillary pressure
     *
     *  \param pC the capillary pressure
     *  \param saturationW the saturation of the wetting phase
     *  \param awnparam the parameter of the biquadratic awn surface curve
     *  \return the derivative \f$\text{d}a_{wn}/\text{d}p_\text{c}\f$
     */
    virtual double dawndpC (double saturationW, double pC, const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi) const
    {
        const std::vector<double> awnparam = soil.awnParam(x, e, xi);
        double value;
        value = awnparam[2] + awnparam[3]*saturationW + 2*awnparam[5]*pC;
        return value;
    }
    /*! \brief the derivative of specific interfacial area function w.r.t. saturation
     *
     *  \param pC the capillary pressure
     *  \param saturationW the saturation of the wetting phase
     *  \param awnparam the parameter of the biquadratic awn surface curve
     *  \return the derivative \f$\text{d}a_{wn}/\text{d}S_w\f$
     */

    virtual double dawndS (double saturationW, double pC, const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi) const
    {
        const std::vector<double> awnparam = soil.awnParam(x, e, xi);
        double value;
        value = awnparam[1] + awnparam[3]*pC + 2*awnparam[4]*saturationW;
        return value;
    }

    /*! \brief wetting phase relative permeability saturation relationship
     *
     *  \param saturationW the saturation of the wetting phase
     *  \return the wetting phase relative permeability
     */
    virtual double krw (const double saturationW, const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi, double T=283.15) const
    {
        switch (soil.relPermFlag(x, e, xi))
            {
            case 0:
                return linearlaw.krw(saturationW, x, e, xi, T);
            case 1:
                return brookscorey.krw(saturationW, x, e, xi, T);
            case 2:
                return vangenuchten.krw(saturationW, x, e, xi, T);
            case 3:
                return auxiliary1.krw(saturationW, x, e, xi, T);
            case 4:
                return auxiliary2.krw(saturationW, x, e, xi, T);
            case 5:
                return auxiliary3.krw(saturationW, x, e, xi, T);
            default:
                DUNE_THROW(NotImplemented, "Matrix2p::modelFlag " << soil.relPermFlag(x, e, xi) << " for TwoPhaseRelations::krw");
            }
    }

    /*! \brief nonwetting phase relative permeability saturation relationship
     *
     *  \param saturationN the saturation of the nonwetting phase
     *  \return the nonwetting phase relative permeability
     */
    virtual double krn (const double saturationN, const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi, double T=283.15) const
    {
        switch (soil.relPermFlag(x, e, xi))
            {
            case 0:
                return linearlaw.krn(saturationN, x, e, xi, T);
            case 1:
                return brookscorey.krn(saturationN, x, e, xi, T);
            case 2:
                return vangenuchten.krn(saturationN, x, e, xi, T);
            case 3:
                return auxiliary1.krn(saturationN, x, e, xi, T);
            case 4:
                return auxiliary2.krn(saturationN, x, e, xi, T);
            case 5:
                return auxiliary3.krn(saturationN, x, e, xi, T);
            default:
                DUNE_THROW(NotImplemented, "Matrix2p::modelFlag " << soil.relPermFlag(x, e, xi) << " for TwoPhaseRelations::krn");
            }
    }

    virtual std::vector<double> kr (const double saturationW, const FieldVector<DT,n>& x, const Entity& e, const FieldVector<DT,n>& xi, double T=283.15) const
    {
        switch (soil.relPermFlag(x, e, xi))
            {
            case 0:
                return linearlaw.kr(saturationW, x, e, xi, T);
            case 1:
                return brookscorey.kr(saturationW, x, e, xi, T);
            case 2:
                return vangenuchten.kr(saturationW, x, e, xi, T);
            case 3:
                return auxiliary1.kr(saturationW, x, e, xi, T);
            case 4:
                return auxiliary2.kr(saturationW, x, e, xi, T);
            case 5:
                return auxiliary3.kr(saturationW, x, e, xi, T);
            default:
                DUNE_THROW(NotImplemented, "Matrix2p::modelFlag " << soil.relPermFlag(x, e, xi) << " for TwoPhaseRelations::kr");
            }
    }

};
}

#endif
