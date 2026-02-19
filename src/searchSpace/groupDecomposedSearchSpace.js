import { TrialState } from '../core/enums.js'
import { SearchSpaceGroup } from './searchSpaceGroup.js'

export class GroupDecomposedSearchSpace {
  constructor(includePruned = false) {
    this.searchSpace = new SearchSpaceGroup()
    this.includePruned = includePruned
  }

  calculate(study, useCache = false) {
    const states = this.includePruned
      ? [TrialState.COMPLETE, TrialState.PRUNED]
      : [TrialState.COMPLETE]

    for (const trial of study.getTrials({ states, useCache })) {
      this.searchSpace.addDistributions(trial.distributions)
    }

    return this.searchSpace.clone()
  }
}
