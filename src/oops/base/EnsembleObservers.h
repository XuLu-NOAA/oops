/*
 * (C) Copyright 2026- UCAR.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "eckit/config/Configuration.h"
#include "oops/base/Departures.h"
#include "oops/base/Geometry.h"
#include "oops/base/IncrementSet.h"
#include "oops/base/LinearModel.h"
#include "oops/base/Model.h"
#include "oops/base/ObsAuxControls.h"
#include "oops/base/ObsEnsemble.h"
#include "oops/base/ObsErrors.h"
#include "oops/base/Observations.h"
#include "oops/base/Observers.h"
#include "oops/base/ObserversTLAD.h"
#include "oops/base/ObsSpaces.h"
#include "oops/base/State.h"
#include "oops/base/StateSet.h"
#include "oops/base/PostProcessor.h"
#include "oops/base/PostProcessorTLAD.h"
#include "oops/base/TrajectorySaver.h"
#include "oops/generic/PseudoLinearModelIncrement4D.h"
#include "oops/generic/PseudoModelState4D.h"
#include "oops/interface/ModelAuxControl.h"
#include "oops/interface/ObsDataVector.h"
#include "oops/util/abor1_cpp.h"
#include "oops/util/Logger.h"

namespace oops {
  class Variables;

/// \brief Ensemble observers
template <typename MODEL, typename OBS>
class EnsembleObservers {
  typedef Observers<MODEL, OBS>       Observers_;
  typedef ObserversTLAD<MODEL, OBS>   ObserversTLAD_;
  typedef Departures<OBS>             Departures_;
  typedef Geometry<MODEL>             Geometry_;
  typedef IncrementSet<MODEL>         IncrementSet_;
  typedef ObsAuxControls<OBS>         ObsAux_;
  typedef ObsAuxIncrements<OBS>       ObsAuxInc_;
  typedef ObsDataVector<OBS, int>     ObsDataInt_;
  typedef ObsEnsemble<OBS>            ObsEnsemble_;
  typedef ObsError<OBS>               ObsError_;
  typedef ObsErrors<OBS>              ObsErrors_;
  typedef Observations<OBS>           Observations_;
  typedef ObsSpaces<OBS>              ObsSpaces_;
  typedef PseudoModelState4D<MODEL>   PseudoModel_;
  typedef PseudoLinearModelIncrement4D<MODEL> PseudoLinearModel_;
  typedef State<MODEL>                State_;
  typedef StateSet<MODEL>             StateSet_;
  typedef State4D<MODEL>              State4D_;
  typedef Increment<MODEL>            Increment_;
  typedef Increment4D<MODEL>          Increment4D_;
  typedef LinearModel<MODEL>          LinearModel_;
  typedef Model<MODEL>                Model_;
  typedef ModelAuxControl<MODEL>      ModelAux_;
  typedef ModelAuxIncrement<MODEL>    ModelAuxInc_;
  typedef ObsDataVector<OBS, int>     ObsData_;
  typedef std::vector<std::shared_ptr<ObsData_>> ObsDataVec_;

 public:
  static const std::string classname() {return "oops::EnsembleObservers";}

  /// initialize observers with \p obspaces, \p geometry, full \p config and \p nens ensemble size
  EnsembleObservers(ObsSpaces_ & obspaces, const Geometry_ & geometry,
                    const eckit::Configuration & config, size_t nens);
  ~EnsembleObservers() = default;

  /// computes ensemble H(\p xx), returns mean H(\p xx), saves as hofx \p iteration
  void computeHofX(const StateSet_ & xx, const StateSet_ & xxmean, size_t iteration);

 protected:
  const Geometry_  & geometry_;   ///< Geometry associated with the updated states
  const ObsSpaces_ & obspaces_;   ///< ObsSpaces used in the update
  std::unique_ptr<ObsErrors_>  R_;         ///< observation errors, set in computeHofX method
  std::unique_ptr<Departures_> invVarR_;   ///< inverse observation error variance for assimilated
                                           ///< observations; set in initializeAssimilatedMask
  std::vector<ObsDataInt_> qcflags_;  ///< quality control flags
  std::unique_ptr<ObserversTLAD_> linear_hofx_;  ///< linear observer
  const eckit::LocalConfiguration obsconf_;  ///< configuration for observations
  const eckit::LocalConfiguration observersconf_;  ///< configuration for observations.observers

  /// Create a mask that excludes observations which will not be assimilated (e.g. failed QC) using
  /// a single ensemble member.
  void initializeAssimilatedMask() {
    // Inverse variances have missing values where obs have failed QC, though R_
    // is only valid for a single ensemble member
    invVarR_ = std::make_unique<Departures_>(R_->inverseVariance());
  }
  /// Update departures for assimilated observations by masking with \p mask
  /// (e.g. nonzero QC flags or missing obs) - should only be done in the
  /// computeHofX method.
  void updateAssimilatedMask(const Departures_ & mask) { invVarR_->mask(mask); }
  /// Apply the assimilated mask to a departures vector \p dep - missing values
  /// in the mask will become missing in the departures.
  void applyAssimilatedMask(Departures_ & dep) const { dep.mask(*invVarR_); }
  /// Apply the non-linear observation operator to the background state \p xx and store the result
  /// in \p yy. If \ref useLinearObserver() returns true, the observation operator is also
  /// linearized about the background state \p xx (cf \ref applyLinearToPerturbations).
  void computeHofX4D(const eckit::Configuration & config, const StateSet_ & xx, Observations_ & yy,
                     const util::Duration & flength, const util::Duration & default_tstep,
                     const ObsAux_ & obsaux, const ModelAux_ & moderr,
                     ObsErrors_ & R, std::vector<ObsDataInt_> & qcflags);
  /// Runs a linear model on 4D perturbations from the ensemble mean ( \p xx - \ref xxmean_ ) and
  /// applies a linearized observation operator \ref linear_hofx_ to background departures in the
  /// ensemble \ref Yb_ at ensemble index \p iens .
  /// Intended for following the procedure from Shlyaeva, A., & Whitaker, J. S. (2018)
  /// https://doi.org/10.1029/2018MS001309 where the linearized observation operator is applied to
  /// ensemble perturbations. In more detail, the obs operator is linearized about the mean state
  /// (H^tilde in eq. 6) - this happens in \ref computeHofX4D when \ref useLinearObserver() returns
  /// true. This is applied to the ensemble perturbations from the mean (X) (Eq 6.). The result is
  /// added to the application of the nonlinear observation operator to the mean state (Eq. 5).
  void applyLinearToPerturbations(const Increment4D_ & dx, const util::Duration & flength,
                                  const util::Duration & default_tstep,
                                  const ObsAuxInc_ & obsauxinc, const ModelAuxInc_ & moderrinc,
                                  const PostProcessor<Increment_> & posttl,
                                  const PostProcessorTLAD<MODEL> & posttrajtl,
                                  Departures_ & tmpDeps);
  bool useLinearObserver_;
};

// -----------------------------------------------------------------------------

template <typename MODEL, typename OBS>
EnsembleObservers<MODEL, OBS>::EnsembleObservers(ObsSpaces_ & obspaces,
                                                 const Geometry_ & geometry,
                                                 const eckit::Configuration & config,
                                                  size_t nens)
  : geometry_(geometry),
    obspaces_(obspaces),
    obsconf_(config, "observations"),
    observersconf_(obsconf_, "observers") {
  // initialize and print options

  useLinearObserver_ = config.getBool("local ensemble DA.use linear observer", false);
  for (size_t jj = 0; jj < obspaces_.size(); ++jj) {
    ObsDataInt_ qcflags(obspaces_[jj], obspaces_[jj].obsvariables());
    qcflags_.push_back(qcflags);
  }
}

// -----------------------------------------------------------------------------

template <typename MODEL, typename OBS>
void EnsembleObservers<MODEL, OBS>::computeHofX4D(const eckit::Configuration & config,
                                                  const StateSet_ & xx, Observations_ & yy,
                                                  const util::Duration & flength,
                                                  const util::Duration & default_tstep,
                                                  const ObsAux_ & obsaux,
                                                  const ModelAux_ & moderr,
                                                  ObsErrors_ & Rmat,
                                                  std::vector<ObsDataInt_> & qcflags) {
  // Setup pseudo model to run on ensemble mean
  State_ init_xx = xx[0];
  std::unique_ptr<PseudoModel_> pseudomodel(new PseudoModel_(xx, default_tstep));
  const Model_ model(std::move(pseudomodel));

  // setup nonlinear postprocessor nonlinear observers
  PostProcessor<State_> post;
  Observers_ hofx(this->obspaces_, this->obsconf_);

  // initialize nonlinear model postprocessor
  hofx.initialize(this->geometry_, obsaux, Rmat, post, config);

  if (useLinearObserver_) {
    // Set up linear observer
    linear_hofx_ = std::make_unique<ObserversTLAD_>(obspaces_, obsconf_);
    // Set up linear postprocessor
    PostProcessorTLAD<MODEL> posttraj;
    // add linearized H(x) to the nonlinear model postprocessor
    linear_hofx_->initializeTraj(this->geometry_, obsaux, posttraj);
    // create TrajectorySaver with hofx_linear, and enroll in post
    post.enrollProcessor(new TrajectorySaver<MODEL>(eckit::LocalConfiguration(),
                                                    this->geometry_, posttraj));
  }

  // run nonlinear model on the ensemble mean
  model.forecast(init_xx, moderr, flength, post);

  // compute nonlinear H(x)
  hofx.finalize(yy, qcflags);

  if (useLinearObserver_) {
    // set the background state to linearise H about.
    linear_hofx_->finalizeTraj(qcflags);
  }
}
// -----------------------------------------------------------------------------

template <typename MODEL, typename OBS>
void EnsembleObservers<MODEL, OBS>::applyLinearToPerturbations(
    const Increment4D_ & dx, const util::Duration & flength, const util::Duration & default_tstep,
    const ObsAuxInc_ & obsauxinc, const ModelAuxInc_ & moderrinc,
    const PostProcessor<Increment_> & posttl, const PostProcessorTLAD<MODEL> & posttrajtl,
    Departures_ & tmpDeps) {
  Increment_ init_dx = dx[0];
  std::unique_ptr<PseudoLinearModel_> pseudolinearmodel =
        std::make_unique<PseudoLinearModel_>(dx, default_tstep);
  const LinearModel_ linear_model(std::move(pseudolinearmodel));
  // run linear model on the ensemble perturbation, compute linear H*dx
  linear_model.forecastTL(init_dx, moderrinc, flength, posttl, posttrajtl);
  linear_hofx_->finalizeTL(obsauxinc, tmpDeps);
}

// -----------------------------------------------------------------------------
template <typename MODEL, typename OBS>
void EnsembleObservers<MODEL, OBS>::computeHofX(const StateSet_ & ens_xx, const StateSet_ & xxmean,
                                                size_t iteration) {
  util::Timer timer(classname(), "computeHofX");

  const size_t nens = ens_xx.ens_size();
  ObsEnsemble_ obsens(obspaces_, nens);
  Observations_ y_mean_xb(obspaces_);

  // Initialize R_ anew for each iteration
  R_.reset(new ObsErrors_(this->observersconf_, this->obspaces_));

  // compute and save H(x)
  std::vector<util::DateTime> times =
      useLinearObserver_ ? ens_xx.times() : xxmean.times();
  util::Duration flength = times[times.size() - 1] - times[0];
  // default_tstep = 2*observation window is passed to PseudoModel as the default
  // pseudomodel time step. It is only used when StateSet has a single state, to enable
  // processing of all observations in the specified window regardless of where in
  // the time window the state is. Observations in
  // ( max(winbgn, xx.time - tstep/2); min(winend, xx.time + tstep/2) ] are
  // processed in H(x).
  util::Duration default_tstep = (obspaces_.windowEnd() - obspaces_.windowStart()) * 2;
  const ModelAux_ moderr(geometry_, eckit::LocalConfiguration());
  const ModelAuxInc_  moderrinc(geometry_, eckit::LocalConfiguration());
  const ObsAux_  obsaux(obspaces_, observersconf_);
  const ObsAuxInc_  obsauxinc(obspaces_, observersconf_);

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
  computeHofX4D(config, xxmean, y_mean_xb, flength, default_tstep, obsaux, moderr,
                *R_, qcflags_);
  y_mean_xb.save("hofx_y_mean_xb"+std::to_string(iteration));

  // QC flags and Obs errors are set to that of the H(mean(Xb))
  R_->save("ObsError");
  initializeAssimilatedMask();

  // save hofx means that hofx will be written out into ObsSpace;
  // if run computeHofX4D several times with save hofx on,
  // the hofx will be overwritten,
  // unless each time specifying iteration differently in the passed config.
  config.set("save hofx", false);
  config.set("save qc", false);
  config.set("save obs errors", false);
  config.set("save obs bias", false);

  if (useLinearObserver_) {
    // add linearized H(x) to the linear model postprocessor
    linear_hofx_->initializeTL(posttrajtl);
  }
  Departures_ tmpDeps(this->obspaces_);

  // use temporary objects for QC flags and obs errors for ensemble members
  // to avoid overwriting the ones from the H(mean(Xb)) calculation
  std::vector<ObsDataInt_> qcflags;
  for (size_t jobs = 0; jobs < obspaces_.size(); ++jobs) {
    ObsDataInt_ flags(obspaces_[jobs], obspaces_[jobs].obsvariables());
    qcflags.push_back(flags);
  }
  ObsErrors_ Rmat(observersconf_, obspaces_);

  for (size_t jj = 0; jj < nens; ++jj) {
    if (useLinearObserver_) {
      // Setup PseudoLinearModelIncrement4D to run on ensemble perturbation
      Increment4D_ dx(geometry_, ens_xx.variables(), times);
      for (size_t it = 0; it < times.size(); ++it) {
        dx(it, 0).diff(ens_xx(it, jj), xxmean[it]);
      }

      // Approximate H(x_i) (obsens[jj]) around the ensemble mean using linearized model and
      // linearized observer. Firstly, apply the linearized obs operator to this ensemble member's
      // background perturbation from the ensemble mean.
      applyLinearToPerturbations(dx, flength, default_tstep, obsauxinc, moderrinc, posttl,
                                 posttrajtl, tmpDeps);
      // Secondly, add this to the ensemble mean in observation space (calculated with the
      // nonlinear obs operator) giving the approximate H(x_i)
      obsens[jj] = y_mean_xb;
      obsens[jj] += tmpDeps;
    } else {
      // These are recalculated for each ensemble member
      times = ens_xx.times();
      flength = times[times.size()-1] - times[0];
      default_tstep = (obspaces_.windowEnd() - obspaces_.windowStart()) * 2;

      const ModelAux_ moderr(geometry_, eckit::LocalConfiguration());
      const ObsAux_  obsaux(obspaces_, observersconf_);

      // Construct a single-member StateSet and populate it from ens_xx
      StateSet_ member_xx(geometry_, ens_xx.variables(), times, ens_xx.commTime());
      for (size_t it = 0; it < times.size(); ++it) {
        member_xx(it, 0) = ens_xx(it, jj);
      }

      computeHofX4D(config, member_xx, obsens[jj], flength, default_tstep,
                    obsaux, moderr, Rmat, qcflags);
    }
    Log::test() << "H(x) for member " << jj+1 << ":" << std::endl << obsens[jj] << std::endl;
    obsens[jj].save("hofx"+std::to_string(iteration)+"_"+std::to_string(jj+1));
  }

  // calculate H(x) ensemble mean
  Observations_ yb_mean(obsens.mean());
  const std::string type = iteration == 0 ? "background" : "analysis";
  const std::string group = iteration == 0 ? "ombg" : "oman";
  Log::test() << "H(x) ensemble " << type << " mean: " << std::endl << yb_mean << std::endl;
  Observations_ yobs(obspaces_, "ObsValue");
  Departures_ diff(yobs - yb_mean);

  diff.save(group);
  Log::test() << type << " y - H(x): " << std::endl << diff << std::endl;
  // display overall background/analysis RMS stats
  Log::test() << group << " RMS: " << diff.rms() << std::endl;
}

// -----------------------------------------------------------------------------

}  // namespace oops