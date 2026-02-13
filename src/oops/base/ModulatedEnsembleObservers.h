/*
 * (C) Copyright 2026- UCAR.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#pragma once

#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "eckit/config/LocalConfiguration.h"
#include "oops/assimilation/ETKFLinearAlgebra.h"
#include "oops/assimilation/gletkfInterface.h"
#include "oops/assimilation/LocalEnsembleSolver.h"
#include "oops/base/Departures.h"
#include "oops/base/DeparturesEnsemble.h"
#include "oops/base/Geometry.h"
#include "oops/base/IncrementSet.h"
#include "oops/base/ObsEnsemble.h"
#include "oops/base/ObsErrors.h"
#include "oops/base/Observations.h"
#include "oops/base/ObsLocalizations.h"
#include "oops/base/ObsSpaces.h"
#include "oops/base/StateSet.h"
#include "oops/generic/VerticalLocEV.h"
#include "oops/interface/GeometryIterator.h"
#include "oops/interface/ObsDataVector.h"
#include "oops/util/ConfigFunctions.h"
#include "oops/util/Logger.h"
#include "oops/util/printRunStats.h"
#include "oops/util/Timer.h"

namespace oops {
  class Variables;

template <typename MODEL, typename OBS>
class ModulatedEnsembleObservers : public EnsembleObservers<MODEL, OBS> {
  typedef Departures<OBS>             Departures_;
  typedef DeparturesEnsemble<OBS>     DeparturesEnsemble_;
  typedef Geometry<MODEL>             Geometry_;
  typedef GeometryIterator<MODEL>     GeometryIterator_;
  typedef Increment<MODEL>            Increment_;
  typedef Increment4D<MODEL>          Increment4D_;
  typedef IncrementSet<MODEL>         IncrementSet_;
  typedef LinearModel<MODEL>          LinearModel_;
  typedef Model<MODEL>                Model_;
  typedef ModelAuxControl<MODEL>      ModelAux_;
  typedef ModelAuxIncrement<MODEL>    ModelAuxInc_;
  typedef ObsAuxControls<OBS>         ObsAux_;
  typedef ObsAuxIncrements<OBS>       ObsAuxInc_;
  typedef ObsDataVector<OBS, int>     ObsDataInt_;
  typedef ObsEnsemble<OBS>            ObsEnsemble_;
  typedef ObsError<OBS>               ObsError_;
  typedef ObsErrors<OBS>              ObsErrors_;
  typedef Observations<OBS>           Observations_;
  typedef Observers<MODEL, OBS>       Observers_;
  typedef ObserversTLAD<MODEL, OBS>   ObserversTLAD_;
  typedef ObsLocalizations<MODEL, OBS> ObsLocalizations_;
  typedef ObsSpaces<OBS>              ObsSpaces_;
  typedef PseudoModelState4D<MODEL>   PseudoModel_;
  typedef PseudoLinearModelIncrement4D<MODEL> PseudoLinearModel_;
  typedef State<MODEL>                State_;
  typedef StateSet<MODEL>             StateSet_;
  typedef VerticalLocEV<MODEL>        VerticalLocEV_;

 public:
  static const std::string classname() {return "oops::DeterministicGETKF";}

  /// Constructor (allocates Wa, wa, HZb_,
  /// saves options from the config, computes VerticalLocEV_)
  ModulatedEnsembleObservers(ObsSpaces_ &,
                     const Geometry_ &,
                     const eckit::Configuration &,
                     size_t,
                     const StateSet_ &,
                     const Variables &);

  void computeHofX(const StateSet_ &, const StateSet_ &, size_t);
 protected:
  VerticalLocEV_ vertloc_;
  const size_t neig_, nens_;
  const Geometry_ & geometry_;
  const Variables incvars_;
  const ObsSpaces_ & obspaces_;
};

// -----------------------------------------------------------------------------

template <typename MODEL, typename OBS>
ModulatedEnsembleObservers<MODEL, OBS>::ModulatedEnsembleObservers(ObsSpaces_ & obspaces,
                                                   const Geometry_ & geometry,
                                                   const eckit::Configuration & config, size_t nens,
                                                   const StateSet_ & xbmean,
                                                   const Variables & incvars)
  : EnsembleObservers<MODEL, OBS>(obspaces, geometry, config, nens),
    vertloc_(config.getSubConfiguration("local ensemble DA.vertical localization"), xbmean[0],
             incvars),
    neig_(vertloc_.neig()), geometry_(geometry), nens_(nens), incvars_(incvars), obspaces_(obspaces) {
  Log::trace() << "ModulatedEnsembleObservers<MODEL, OBS>::create starting" << std::endl;
  Log::trace() << "ModulatedEnsembleObservers<MODEL, OBS>::create done" << std::endl;
}

// -----------------------------------------------------------------------------
template <typename MODEL, typename OBS>
void ModulatedEnsembleObservers<MODEL, OBS>::computeHofX(const StateSet_ & ens_xx,
                                                         const StateSet_ & xbmean,
                                                         size_t iteration) {
  util::Timer timer(classname(), "computeHofX");

  ModelAux_ moderr(geometry_, eckit::LocalConfiguration());
  ModelAuxInc_  moderrinc(geometry_, eckit::LocalConfiguration());
  ObsAux_  obsaux(this->obspaces_, this->observersconf_);
  ObsAuxInc_  obsauxinc(this->obspaces_, this->observersconf_);

  Observations_ yb_mean(this->obspaces_);
  Departures_ tmpDeps(this->obspaces_);

  // Initialize R_ anew for each iteration
  oops::Log::info() << " before create obserrors" << std::endl;
  this->R_.reset(new ObsErrors_(this->observersconf_, this->obspaces_));

  oops::Log::info() << " after create obserrors" << std::endl;
  const util::Duration default_tstep = (this->obspaces_.windowEnd()
                                      - this->obspaces_.windowStart()) * 2;
  const std::vector<util::DateTime> times = ens_xx.times();
  const util::Duration flength = times[times.size()-1] - times[0];

  Increment4D_ dx(geometry_, incvars_, times);
  std::vector<int> evMembers(neig_);
  std::iota(evMembers.begin(), evMembers.end(), 0);
  IncrementSet_ Ztmp(geometry_, incvars_, times, xbmean.commTime(), evMembers);

  size_t ii = 0;
  if (this->useLinearObserver_) {
    // set up postprocessors for the linear model run on ensemble perturbations
    PostProcessor<Increment_> posttl;
    PostProcessorTLAD<MODEL> posttrajtl;

    // save QC filters, obs bias, ob errors for the mean (H(x) is saved separately)
    eckit::LocalConfiguration config;
    config.set("save hofx", false);
    config.set("save qc", true);
    config.set("save obs errors", true);
    config.set("save obs bias", true);
    config.set("iteration", std::to_string(iteration));

    this->computeHofX4D(config, xbmean, yb_mean, flength, default_tstep, obsaux, moderr,
                        *this->R_, this->qcflags_);

    // for linear H, yb_mean==y_mean_xb
    Observations_ y_mean_xb(yb_mean);
    y_mean_xb.save("hofx_y_mean_xb"+std::to_string(iteration));

    // QC flags and Obs errors are set to that of the H(mean(Xb))
    this->R_->save("ObsError");
    this->initializeAssimilatedMask();

    // calculate obs departures
    Observations_ yobs(this->obspaces_, "ObsValue");
    Departures omb = yobs - yb_mean;
    // Need to mask out any missing departures as well as those that have failed QC
    this->updateAssimilatedMask(omb);

    // add linearized H(x) to the linear model postprocessor
    this->linear_hofx_->initializeTL(posttrajtl);

    for (size_t iens = 0; iens < nens_; ++iens) {
      Log::info() << " DeterministicGETKF::computeHofX starting ensemble member "
                  << iens+1 << std::endl;
      util::printRunStats("DeterministicGETKF calculate hofx");
      Log::info() << " GETKFSolver::computeHofX starting ensemble member " << iens+1 << std::endl;
      util::printRunStats("GETKFSolver calculate hofx");
      tmpDeps.zero();
      // Setup PseudoLinearModelIncrement4D to run on ensemble perturbation
      for (size_t it = 0; it < times.size(); ++it) {
        dx(it, 0).diff(ens_xx(it, iens), xbmean[it]);
      }
      // Approximate H(x) using linearized model and linearized observer
      this->applyLinearToPerturbations(dx, flength, default_tstep, obsauxinc, moderrinc,
                                    posttl, posttrajtl, tmpDeps);
      Observations_ tmpObs(yb_mean);
      tmpObs += tmpDeps;
      Log::test() << "H(x) for member " << iens+1 << ":" << std::endl << tmpObs << std::endl;
      tmpObs.save("hofx"+std::to_string(iteration)+"_"+std::to_string(iens+1));

      // observe modulated members
      vertloc_.modulateIncrement(dx, Ztmp);
      for (size_t ieig = 0; ieig < neig_; ++ieig) {
        Increment4D_ z_eig(geometry_, incvars_, times);
        for (size_t it = 0; it < times.size(); ++it) {
          z_eig[it] = Ztmp(it, ieig);
        }
        this->applyLinearToPerturbations(z_eig, flength, default_tstep, obsauxinc, moderrinc,
                                         posttl, posttrajtl, tmpDeps);
        Observations_ tmpObs(yb_mean);
        tmpObs += tmpDeps;
        tmpObs.save("hofxm"+std::to_string(iteration)+"_"+std::to_string(ieig+1)+
                    "_"+std::to_string(iens+1));
        ii = ii + 1;
      }
    }
  } else {
    oops::Log::info() << " computeHofX for mean state" << std::endl;
    // compute and save H(x)
    std::vector<util::DateTime> times = xbmean.times();
    util::Duration flength = times[times.size() - 1] - times[0];
    util::Duration default_tstep = (obspaces_.windowEnd() - obspaces_.windowStart()) * 2;
    const ModelAux_ moderr(geometry_, eckit::LocalConfiguration());
    const ModelAuxInc_  moderrinc(geometry_, eckit::LocalConfiguration());
    const ObsAux_  obsaux(obspaces_, this->observersconf_);
    const ObsAuxInc_  obsauxinc(obspaces_, this->observersconf_);

    // set up postprocessors for the linear model run on ensemble perturbations
    PostProcessor<Increment_> posttl;
    PostProcessorTLAD<MODEL> posttrajtl;

    // save QC filters, obs bias, ob errors for H(mean(Xb) (H(x_i) are saved separately)
    eckit::LocalConfiguration config;
    config.set("save hofx", false);
    config.set("save qc", true);
    config.set("save obs errors", true);
    config.set("save obs bias", true);
    config.set("iteration", std::to_string(iteration));

    // run the model forecast on the background ensemble mean and optionally linearise the observer
    // about this trajectory. save obs errors and qc flags for H(mean(Xb))
    this->computeHofX4D(config, xbmean, yb_mean, flength, default_tstep, obsaux, moderr,
                  *(this->R_), this->qcflags_);
    yb_mean.save("hofx_y_mean_xb"+std::to_string(iteration));

    // QC flags and Obs errors are set to that of the H(mean(Xb))
    this->R_->save("ObsError");
    this->initializeAssimilatedMask();

    // save hofx means that hofx will be written out into ObsSpace;
    // if run computeHofX4D several times with save hofx on,
    // the hofx will be overwritten,
    // unless each time specifying iteration differently in the passed config.
    config.set("save hofx", false);
    config.set("save qc", false);
    config.set("save obs errors", false);
    config.set("save obs bias", false);

    Departures_ tmpDeps(this->obspaces_);

    // use temporary objects for QC flags and obs errors for ensemble members
    // to avoid overwriting the ones from the H(mean(Xb)) calculation
    std::vector<ObsDataInt_> qcflags;
    for (size_t jobs = 0; jobs < obspaces_.size(); ++jobs) {
      ObsDataInt_ flags(obspaces_[jobs], obspaces_[jobs].obsvariables());
      qcflags.push_back(flags);
    }
    ObsErrors_ Rmat(this->observersconf_, obspaces_);

    Observations_ tmpObs(this->obspaces_);
    ObsEnsemble_ obsens(obspaces_, nens_);
    for (size_t iens = 0; iens < nens_; ++iens) {
      // These are recalculated for each ensemble member
      times = ens_xx.times();
      flength = times[times.size()-1] - times[0];
      default_tstep = (obspaces_.windowEnd() - obspaces_.windowStart()) * 2;

      const ModelAux_ moderr(geometry_, eckit::LocalConfiguration());
      const ObsAux_  obsaux(obspaces_, this->observersconf_);

      // Construct a single-member StateSet and populate it from ens_xx
      StateSet_ member_xx(geometry_, ens_xx.variables(), times, ens_xx.commTime());
      for (size_t it = 0; it < times.size(); ++it) {
        member_xx(it, 0) = ens_xx(it, iens);
      }

      this->computeHofX4D(config, member_xx, obsens[iens], flength, default_tstep,
                    obsaux, moderr, Rmat, qcflags);
      Log::test() << "H(x) for member " << iens+1 << ":" << std::endl << obsens[iens] << std::endl;
      obsens[iens].save("hofx"+std::to_string(iteration)+"_"+std::to_string(iens+1));

      Log::info() << " DeterministicGETKF::computeHofX starting ensemble member "
                  << iens+1 << std::endl;
      util::printRunStats("DeterministicGETKF calculate hofx");
      for (size_t it = 0; it < times.size(); ++it) {
        dx(it, 0).diff(ens_xx(it, iens), xbmean[it]);
      }
      vertloc_.modulateIncrement(dx, Ztmp);
      for (size_t ieig = 0; ieig < neig_; ++ieig) {
        StateSet_ tmpState = xbmean;
        for (size_t it = 0; it < times.size(); ++it) {
          tmpState[it] += Ztmp(it, ieig);
        }
       
        this->computeHofX4D(config, tmpState, tmpObs, flength, default_tstep, obsaux, moderr,
                            Rmat, qcflags);
        // mask H(x) ensemble perturbations - i.e. make sure that obs that have
        // failed QC on one ensemble member fail for all
        Departures_ tmpDep = tmpObs - yb_mean;
        this->updateAssimilatedMask(tmpDep);
        tmpObs.save("hofxm"+std::to_string(iteration)+"_"+std::to_string(ieig+1)+
                      "_"+std::to_string(iens+1));
        ii = ii + 1;
      }
    }
    yb_mean = obsens.mean();
  }
  // calculate H(x) ensemble mean
  const std::string type = iteration == 0 ? "background" : "analysis";
  const std::string group = iteration == 0 ? "ombg" : "oman";
  Log::test() << "H(x) ensemble " << type << " mean: " << std::endl << yb_mean << std::endl;
  Observations_ yobs(this->obspaces_, "ObsValue");
  Departures_ diff(yobs - yb_mean);

  diff.save(group);
  Log::test() << type << " y - H(x): " << std::endl << diff << std::endl;
  // display overall background/analysis RMS stats
  Log::test() << group << " RMS: " << diff.rms() << std::endl;

}

}  // namespace oops