import { TrialState } from '../core/enums.js'
import { shallowCopy, sortObjectEntries } from '../core/objectUtils.js'
import { isFinishedState } from '../study/trialStateUtils.js'

export class IntersectionSearchSpace {
  constructor(includePruned = false) {
    this.includePruned = includePruned
    this.cachedTrialNumber = -1
    this.searchSpace = null
  }

  calculate(study, useCache = false) {
    const statesOfInterest = [TrialState.COMPLETE, TrialState.WAITING, TrialState.RUNNING]
    if (this.includePruned) {
      statesOfInterest.push(TrialState.PRUNED)
    }

    let nextCachedTrialNumber = -1
    const trials = study.getTrials({ useCache })
    for (let i = trials.length - 1; i >= 0; i -= 1) {
      const trial = trials[i]
      if (!statesOfInterest.includes(trial.state)) {
        continue
      }

      if (nextCachedTrialNumber === -1) {
        nextCachedTrialNumber = trial.number + 1
      }

      if (this.cachedTrialNumber > trial.number) {
        break
      }

      if (!isFinishedState(trial.state)) {
        nextCachedTrialNumber = trial.number
        continue
      }

      if (this.searchSpace === null) {
        this.searchSpace = shallowCopy(trial.distributions)
        continue
      }

      const next = {}
      for (const [name, distribution] of Object.entries(this.searchSpace)) {
        const trialDist = trial.distributions[name]
        if (trialDist && trialDist.equals(distribution)) {
          next[name] = distribution
        }
      }
      this.searchSpace = next
    }

    this.cachedTrialNumber = nextCachedTrialNumber
    const result = this.searchSpace || {}
    const sortedResult = {}
    for (const [name, distribution] of sortObjectEntries(result)) {
      sortedResult[name] = distribution
    }
    return sortedResult
  }
}
