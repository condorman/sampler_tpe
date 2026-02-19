import { TrialState } from '../core/enums.js'

export function isFinishedState(state) {
  return state !== TrialState.RUNNING && state !== TrialState.WAITING
}
