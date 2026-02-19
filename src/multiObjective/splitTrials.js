import { CONSTRAINTS_KEY, EPS } from '../core/constants.js'
import { StudyDirection, TrialState } from '../core/enums.js'
import { computeHypervolume } from './hypervolume.js'
import { getReferencePoint, solveHssp } from './hssp.js'
import { fastNonDominationRank, isParetoFront } from './pareto.js'

export function splitCompleteTrialsSingleObjective(trials, study, nBelow) {
  const sorted = [...trials].sort((a, b) => {
    if (study.direction === StudyDirection.MINIMIZE) {
      return a.value - b.value
    }
    return b.value - a.value
  })
  return [sorted.slice(0, nBelow), sorted.slice(nBelow)]
}

export function splitCompleteTrialsMultiObjective(trials, study, nBelow) {
  if (nBelow === 0) {
    return [[], [...trials]]
  }
  if (nBelow === trials.length) {
    return [[...trials], []]
  }

  const lvals = trials.map((trial) =>
    trial.values.map((value, i) =>
      study.directions[i] === StudyDirection.MAXIMIZE ? -value : value
    )
  )

  const nondominationRanks = fastNonDominationRank(lvals, null, nBelow)
  const rankCounts = new Map()
  for (const r of nondominationRanks) {
    rankCounts.set(r, (rankCounts.get(r) || 0) + 1)
  }
  const ranks = [...rankCounts.keys()].sort((a, b) => a - b)

  let cum = 0
  let lastRankBeforeTiebreak = -1
  for (const r of ranks) {
    cum += rankCounts.get(r)
    if (cum <= nBelow) {
      lastRankBeforeTiebreak = r
    }
  }

  let indicesBelow = []
  for (let i = 0; i < nondominationRanks.length; i += 1) {
    if (nondominationRanks[i] <= lastRankBeforeTiebreak) {
      indicesBelow.push(i)
    }
  }

  if (indicesBelow.length < nBelow) {
    const needRank = lastRankBeforeTiebreak + 1
    const needIndices = []
    for (let i = 0; i < nondominationRanks.length; i += 1) {
      if (nondominationRanks[i] === needRank) {
        needIndices.push(i)
      }
    }

    const rankLossVals = needIndices.map((i) => lvals[i])
    const subsetSize = nBelow - indicesBelow.length
    const selected = solveHssp(
      rankLossVals,
      needIndices,
      subsetSize,
      getReferencePoint(rankLossVals)
    )
    indicesBelow = [...indicesBelow, ...selected]
  }

  const belowSet = new Set(indicesBelow)
  const below = []
  const above = []
  for (let i = 0; i < trials.length; i += 1) {
    if (belowSet.has(i)) {
      below.push(trials[i])
    } else {
      above.push(trials[i])
    }
  }

  return [below, above]
}

export function splitCompleteTrials(trials, study, nBelow) {
  const clipped = Math.min(nBelow, trials.length)
  if (study.directions.length <= 1) {
    return splitCompleteTrialsSingleObjective(trials, study, clipped)
  }
  return splitCompleteTrialsMultiObjective(trials, study, clipped)
}

export function getPrunedTrialScore(trial, study) {
  const entries = Object.entries(trial.intermediate_values)
  if (entries.length > 0) {
    entries.sort((a, b) => Number(a[0]) - Number(b[0]))
    const [stepRaw, val] = entries[entries.length - 1]
    const step = Number(stepRaw)
    if (Number.isNaN(val)) {
      return [-step, Infinity]
    }
    if (study.direction === StudyDirection.MINIMIZE) {
      return [-step, val]
    }
    return [-step, -val]
  }
  return [1, 0]
}

export function splitPrunedTrials(trials, study, nBelow) {
  const clipped = Math.min(nBelow, trials.length)
  const sorted = [...trials].sort((a, b) => {
    const sa = getPrunedTrialScore(a, study)
    const sb = getPrunedTrialScore(b, study)
    if (sa[0] !== sb[0]) return sa[0] - sb[0]
    return sa[1] - sb[1]
  })
  return [sorted.slice(0, clipped), sorted.slice(clipped)]
}

export function getInfeasibleTrialScore(trial) {
  const constraint = trial.system_attrs[CONSTRAINTS_KEY]
  if (constraint === undefined || constraint === null) {
    return Infinity
  }
  let s = 0
  for (const v of constraint) {
    if (v > 0) s += v
  }
  return s
}

export function splitInfeasibleTrials(trials, nBelow) {
  const clipped = Math.min(nBelow, trials.length)
  const sorted = [...trials].sort(
    (a, b) => getInfeasibleTrialScore(a) - getInfeasibleTrialScore(b)
  )
  return [sorted.slice(0, clipped), sorted.slice(clipped)]
}

export function splitTrials(study, trials, nBelow, constraintsEnabled) {
  const complete = []
  const pruned = []
  const running = []
  const infeasible = []

  for (const trial of trials) {
    if (trial.state === TrialState.RUNNING) {
      running.push(trial)
    } else if (constraintsEnabled && getInfeasibleTrialScore(trial) > 0) {
      infeasible.push(trial)
    } else if (trial.state === TrialState.COMPLETE) {
      complete.push(trial)
    } else if (trial.state === TrialState.PRUNED) {
      pruned.push(trial)
    } else {
      throw new Error(`Unexpected trial state in split: ${trial.state}`)
    }
  }

  const [belowComplete, aboveComplete] = splitCompleteTrials(complete, study, nBelow)
  let remaining = Math.max(0, nBelow - belowComplete.length)
  const [belowPruned, abovePruned] = splitPrunedTrials(pruned, study, remaining)
  remaining = Math.max(0, remaining - belowPruned.length)
  const [belowInfeasible, aboveInfeasible] = splitInfeasibleTrials(infeasible, remaining)

  const below = [...belowComplete, ...belowPruned, ...belowInfeasible].sort(
    (a, b) => a.number - b.number
  )
  const above = [...aboveComplete, ...abovePruned, ...aboveInfeasible, ...running].sort(
    (a, b) => a.number - b.number
  )

  return [below, above]
}

export function calculateWeightsBelowForMultiObjective(
  study,
  belowTrials,
  constraintsFunc
) {
  const feasibleMask = belowTrials.map((trial) => {
    if (constraintsFunc === null || constraintsFunc === undefined) {
      return true
    }
    return constraintsFunc(trial).every((c) => c <= 0)
  })

  const weights = feasibleMask.map((ok) => (ok ? 1 : EPS))
  const feasibleIndices = []
  for (let i = 0; i < feasibleMask.length; i += 1) {
    if (feasibleMask[i]) feasibleIndices.push(i)
  }

  if (feasibleIndices.length <= 1) {
    return weights
  }

  const lvals = feasibleIndices.map((idx) => {
    const values = belowTrials[idx].values
    return values.map((v, i) =>
      study.directions[i] === StudyDirection.MAXIMIZE ? -v : v
    )
  })

  const refPoint = getReferencePoint(lvals)
  const onFront = isParetoFront(lvals, false)
  const paretoSols = lvals.filter((_, i) => onFront[i])
  const hv = computeHypervolume(paretoSols, refPoint, true)
  if (!Number.isFinite(hv)) {
    return weights
  }

  const contribsFeasible = new Array(feasibleIndices.length).fill(0)
  if (study.directions.length <= 3) {
    const frontIndices = []
    for (let i = 0; i < onFront.length; i += 1) {
      if (onFront[i]) frontIndices.push(i)
    }

    for (const fi of frontIndices) {
      const leaveOneOut = paretoSols.filter((_, j) => j !== frontIndices.indexOf(fi))
      const hvLoo = computeHypervolume(leaveOneOut, refPoint, true)
      contribsFeasible[fi] = hv - hvLoo
    }
  } else {
    for (let i = 0; i < feasibleIndices.length; i += 1) {
      let p = 1
      for (let d = 0; d < refPoint.length; d += 1) {
        p *= refPoint[d] - lvals[i][d]
      }
      contribsFeasible[i] = p
    }
  }

  let maxContrib = EPS
  for (const c of contribsFeasible) {
    if (c > maxContrib) maxContrib = c
  }

  for (let i = 0; i < feasibleIndices.length; i += 1) {
    const globalIdx = feasibleIndices[i]
    weights[globalIdx] = Math.max(contribsFeasible[i] / maxContrib, EPS)
  }

  return weights
}
