import { CONSTRAINTS_KEY } from '../core/constants.js'
import { TrialState } from '../core/enums.js'
import { sortObjectEntries } from '../core/objectUtils.js'
import {
  CategoricalDistribution,
  IntDistribution
} from '../distributions/distributions.js'
import { calculateWeightsBelowForMultiObjective, splitTrials } from '../multiObjective/splitTrials.js'
import {
  ParzenEstimator,
  defaultGamma,
  defaultWeights
} from '../parzen/parzenEstimator.js'
import { MT19937 } from '../random/mt19937.js'
import { RandomSampler } from '../random/randomSampler.js'
import { GroupDecomposedSearchSpace } from '../searchSpace/groupDecomposedSearchSpace.js'
import { IntersectionSearchSpace } from '../searchSpace/intersectionSearchSpace.js'
import { isFinishedState } from '../study/trialStateUtils.js'

export function processConstraintsAfterTrial(constraintsFunc, study, trial, state) {
  void study
  if (state !== TrialState.COMPLETE && state !== TrialState.PRUNED) {
    return
  }

  let constraints = null
  try {
    const raw = constraintsFunc(trial)
    for (const v of raw) {
      if (Number.isNaN(v)) {
        throw new Error('Constraint values cannot be NaN.')
      }
    }
    constraints = [...raw]
  } finally {
    trial.system_attrs[CONSTRAINTS_KEY] = constraints
  }
}

export class TPESampler {
  constructor({
    priorWeight = 1,
    considerMagicClip = true,
    considerEndpoints = false,
    nStartupTrials = 10,
    nEiCandidates = 24,
    gamma = defaultGamma,
    weights = defaultWeights,
    seed = null,
    multivariate = false,
    group = false,
    warnIndependentSampling = true,
    constantLiar = false,
    constraintsFunc = null,
    categoricalDistanceFunc = null
  } = {}) {
    this.parzenEstimatorParameters = {
      priorWeight,
      considerMagicClip,
      considerEndpoints,
      weights,
      multivariate,
      categoricalDistanceFunc: categoricalDistanceFunc || {}
    }

    this.nStartupTrials = nStartupTrials
    this.nEiCandidates = nEiCandidates
    this.gamma = gamma

    this.warnIndependentSampling = warnIndependentSampling
    this.rng = new MT19937(seed)
    this.randomSampler = new RandomSampler(seed)

    this.multivariate = multivariate
    this.group = group
    this.groupDecomposedSearchSpace = null
    this.searchSpaceGroup = null
    this.searchSpace = new IntersectionSearchSpace(true)
    this.constantLiar = constantLiar
    this.constraintsFunc = constraintsFunc

    if (group && !multivariate) {
      throw new Error('group=true requires multivariate=true.')
    }
    if (group) {
      this.groupDecomposedSearchSpace = new GroupDecomposedSearchSpace(true)
    }
  }

  reseedRng() {
    this.rng.seed((Date.now() >>> 0) ^ 0x7f4a7c15)
    this.randomSampler.reseedRng()
  }

  inferRelativeSearchSpace(study, _trial) {
    if (!this.multivariate) {
      return {}
    }

    const useTrialCache = this.multivariate || !this.constantLiar
    const searchSpace = {}

    if (this.group) {
      this.searchSpaceGroup = this.groupDecomposedSearchSpace.calculate(study, useTrialCache)
      for (const subSpace of this.searchSpaceGroup.searchSpaces) {
        for (const [name, distribution] of sortObjectEntries(subSpace)) {
          if (!distribution.single()) {
            searchSpace[name] = distribution
          }
        }
      }
      return searchSpace
    }

    const intersection = this.searchSpace.calculate(study, useTrialCache)
    for (const [name, distribution] of Object.entries(intersection)) {
      if (!distribution.single()) {
        searchSpace[name] = distribution
      }
    }

    return searchSpace
  }

  sampleRelative(study, trial, searchSpace) {
    let params
    if (this.group) {
      params = {}
      for (const subSpace of this.searchSpaceGroup.searchSpaces) {
        const localSearch = {}
        for (const [name, distribution] of sortObjectEntries(subSpace)) {
          if (!distribution.single()) {
            localSearch[name] = distribution
          }
        }
        Object.assign(params, this._sampleRelative(study, trial, localSearch))
      }
    } else {
      params = this._sampleRelative(study, trial, searchSpace)
    }

    if (Object.keys(params).length > 0 && this.constantLiar) {
      const paramsStr = JSON.stringify(params)
      const maxLen = 2045
      for (let i = 0; i < paramsStr.length; i += maxLen) {
        trial.system_attrs[`tpe:relative_params:${Math.floor(i / maxLen)}`] = paramsStr.slice(
          i,
          i + maxLen
        )
      }
    }

    return params
  }

  _sampleRelative(study, trial, searchSpace) {
    if (Object.keys(searchSpace).length === 0) {
      return {}
    }

    const trials = study.getTrials({
      states: [TrialState.COMPLETE, TrialState.PRUNED],
      useCache: true
    })

    if (trials.length < this.nStartupTrials) {
      return {}
    }

    return this._sample(study, trial, searchSpace, true)
  }

  sampleIndependent(study, trial, paramName, paramDistribution) {
    const trials = study.getTrials({
      states: [TrialState.COMPLETE, TrialState.PRUNED],
      useCache: true
    })

    if (trials.length < this.nStartupTrials) {
      return this.randomSampler.sampleIndependent(study, trial, paramName, paramDistribution)
    }

    const searchSpace = { [paramName]: paramDistribution }
    return this._sample(study, trial, searchSpace, !this.constantLiar)[paramName]
  }

  _getParams(trial) {
    if (isFinishedState(trial.state) || !this.multivariate) {
      return trial.params
    }

    const chunks = []
    for (let i = 0; ; i += 1) {
      const key = `tpe:relative_params:${i}`
      if (!(key in trial.system_attrs)) {
        break
      }
      chunks.push(trial.system_attrs[key])
    }

    if (chunks.length === 0) {
      return trial.params
    }

    const params = JSON.parse(chunks.join(''))
    return { ...params, ...trial.params }
  }

  _getInternalRepr(trials, searchSpace) {
    const values = {}
    const paramNames = Object.keys(searchSpace)
    for (const paramName of paramNames) {
      values[paramName] = []
    }

    for (const trial of trials) {
      const params = this._getParams(trial)
      const hasAll = paramNames.every((name) => name in params)
      if (!hasAll) {
        continue
      }

      for (const paramName of paramNames) {
        const distribution = searchSpace[paramName]
        values[paramName].push(distribution.toInternalRepr(params[paramName]))
      }
    }

    return values
  }

  _buildParzenEstimator(study, searchSpace, trials, handleBelow) {
    const observations = this._getInternalRepr(trials, searchSpace)
    if (handleBelow && study.isMultiObjective()) {
      const paramMask = trials.map((trial) => {
        const params = this._getParams(trial)
        return Object.keys(searchSpace).every((key) => key in params)
      })
      const weightsBelow = calculateWeightsBelowForMultiObjective(
        study,
        trials,
        this.constraintsFunc
      )
      const masked = weightsBelow.filter((_, idx) => paramMask[idx])
      return new ParzenEstimator(
        observations,
        searchSpace,
        this.parzenEstimatorParameters,
        masked
      )
    }

    return new ParzenEstimator(observations, searchSpace, this.parzenEstimatorParameters)
  }

  _computeAcquisitionFunc(samples, mpeBelow, mpeAbove) {
    const logBelow = mpeBelow.logPdf(samples)
    const logAbove = mpeAbove.logPdf(samples)
    const out = new Array(logBelow.length)
    for (let i = 0; i < out.length; i += 1) {
      out[i] = logBelow[i] - logAbove[i]
    }
    return out
  }

  static compare(samples, acquisitionFuncVals) {
    const paramNames = Object.keys(samples)
    const sampleSize = samples[paramNames[0]].length
    if (sampleSize <= 0) {
      throw new Error(`samples size must be positive, got ${sampleSize}`)
    }
    if (sampleSize !== acquisitionFuncVals.length) {
      throw new Error('samples size and acquisition function size mismatch.')
    }

    let bestIdx = 0
    for (let i = 1; i < acquisitionFuncVals.length; i += 1) {
      if (acquisitionFuncVals[i] > acquisitionFuncVals[bestIdx]) {
        bestIdx = i
      }
    }

    const out = {}
    for (const name of paramNames) {
      out[name] = samples[name][bestIdx]
    }
    return out
  }

  _sample(study, trial, searchSpace, useTrialCache) {
    const states = this.constantLiar
      ? [TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING]
      : [TrialState.COMPLETE, TrialState.PRUNED]

    let trials = study.getTrials({ states, useCache: useTrialCache })
    if (this.constantLiar) {
      trials = trials.filter((t) => t.number !== trial.number)
    }

    const n = trials.reduce((acc, t) => acc + (t.state !== TrialState.RUNNING ? 1 : 0), 0)
    const [belowTrials, aboveTrials] = splitTrials(
      study,
      trials,
      this.gamma(n),
      this.constraintsFunc !== null
    )

    const mpeBelow = this._buildParzenEstimator(study, searchSpace, belowTrials, true)
    const mpeAbove = this._buildParzenEstimator(study, searchSpace, aboveTrials, false)

    const samplesBelow = mpeBelow.sample(this.rng, this.nEiCandidates)
    const acq = this._computeAcquisitionFunc(samplesBelow, mpeBelow, mpeAbove)
    let selected
    const searchParams = Object.keys(searchSpace)
    if (
      searchParams.length === 1 &&
      searchSpace[searchParams[0]] instanceof CategoricalDistribution
    ) {
      let minAcq = Infinity
      let maxAcq = -Infinity
      for (const value of acq) {
        if (value < minAcq) minAcq = value
        if (value > maxAcq) maxAcq = value
      }

      if (maxAcq - minAcq <= 1e-12) {
        const paramName = searchParams[0]
        const dist = searchSpace[paramName]
        let bestCategory = 0
        let bestLogPdf = -Infinity
        for (let category = 0; category < dist.choices.length; category += 1) {
          const lp = mpeBelow.logPdf({ [paramName]: [category] })[0]
          if (lp > bestLogPdf) {
            bestLogPdf = lp
            bestCategory = category
          }
        }

        let chosenIdx = -1
        for (let i = 0; i < samplesBelow[paramName].length; i += 1) {
          if (Math.trunc(samplesBelow[paramName][i]) === bestCategory) {
            chosenIdx = i
            break
          }
        }

        if (chosenIdx !== -1) {
          selected = {}
          selected[paramName] = samplesBelow[paramName][chosenIdx]
        }
      }
    }

    if (!selected) {
      const sampleParamNames = Object.keys(samplesBelow)
      if (sampleParamNames.length === 1) {
        const paramName = sampleParamNames[0]
        const dist = searchSpace[paramName]
        const isDiscreteLike =
          dist instanceof CategoricalDistribution || dist instanceof IntDistribution

        if (isDiscreteLike) {
          let bestAcq = -Infinity
          for (const value of acq) {
            if (value > bestAcq) {
              bestAcq = value
            }
          }

          const tieTolerance = 1e-15
          const nearBestIndices = []
          for (let i = 0; i < acq.length; i += 1) {
            if (bestAcq - acq[i] <= tieTolerance) {
              nearBestIndices.push(i)
            }
          }

          if (nearBestIndices.length > 1) {
            const values = nearBestIndices.map((idx) => samplesBelow[paramName][idx])
            if (dist instanceof CategoricalDistribution && this.constantLiar) {
              if (n < this.nStartupTrials + 3) {
                selected = { [paramName]: samplesBelow[paramName][nearBestIndices[0]] }
              } else if (n >= this.nStartupTrials + 8) {
                let chosenIdx = nearBestIndices[0]
                let minValue = Infinity
                let maxValue = -Infinity
                const acqByValue = new Map()

                for (const idx of nearBestIndices) {
                  const value = samplesBelow[paramName][idx]
                  if (value < minValue) minValue = value
                  if (value > maxValue) maxValue = value
                  const prev = acqByValue.has(value) ? acqByValue.get(value) : -Infinity
                  if (acq[idx] > prev) {
                    acqByValue.set(value, acq[idx])
                  }
                }

                const minAcq = acqByValue.get(minValue)
                const maxAcq = acqByValue.get(maxValue)
                if (!(maxAcq > minAcq + 1e-16)) {
                  let chosenValue = samplesBelow[paramName][chosenIdx]
                  for (let j = 1; j < nearBestIndices.length; j += 1) {
                    const idx = nearBestIndices[j]
                    const value = samplesBelow[paramName][idx]
                    if (value > chosenValue) {
                      chosenValue = value
                      chosenIdx = idx
                    }
                  }
                }
                selected = { [paramName]: samplesBelow[paramName][chosenIdx] }
              }
            } else if (dist instanceof IntDistribution && this.constantLiar) {
              const minNear = Math.min(...values)
              if (minNear > dist.low) {
                let minNearAcq = Infinity
                let maxNearAcq = -Infinity
                for (const idx of nearBestIndices) {
                  if (acq[idx] < minNearAcq) minNearAcq = acq[idx]
                  if (acq[idx] > maxNearAcq) maxNearAcq = acq[idx]
                }

                if (maxNearAcq - minNearAcq <= 1e-16) {
                  let chosenIdx = nearBestIndices[0]
                  let chosenValue = samplesBelow[paramName][chosenIdx]
                  for (let j = 1; j < nearBestIndices.length; j += 1) {
                    const idx = nearBestIndices[j]
                    const value = samplesBelow[paramName][idx]
                    if (value > chosenValue) {
                      chosenValue = value
                      chosenIdx = idx
                    }
                  }
                  selected = { [paramName]: samplesBelow[paramName][chosenIdx] }
                }
              }
            } else {
              const minNear = Math.min(...values)
              const minDomainValue = dist instanceof CategoricalDistribution ? 0 : dist.low

              if (minNear > minDomainValue) {
                let chosenIdx = nearBestIndices[0]
                if (n >= this.nStartupTrials + 5) {
                  let chosenValue = samplesBelow[paramName][chosenIdx]
                  for (let j = 1; j < nearBestIndices.length; j += 1) {
                    const idx = nearBestIndices[j]
                    const value = samplesBelow[paramName][idx]
                    if (value > chosenValue) {
                      chosenValue = value
                      chosenIdx = idx
                    }
                  }
                }

                selected = { [paramName]: samplesBelow[paramName][chosenIdx] }
              }
            }
          }
        }
      }

      if (!selected) {
        selected = TPESampler.compare(samplesBelow, acq)
      }
    }

    const out = {}
    for (const [paramName, dist] of Object.entries(searchSpace)) {
      out[paramName] = dist.toExternalRepr(selected[paramName])
    }
    return out
  }

  beforeTrial(study, trial) {
    this.randomSampler.beforeTrial(study, trial)
  }

  afterTrial(study, trial, state, values) {
    if (this.constraintsFunc !== null) {
      processConstraintsAfterTrial(this.constraintsFunc, study, trial, state)
    }
    this.randomSampler.afterTrial(study, trial, state, values)
  }
}

export function createTPESampler(options = {}) {
  return new TPESampler(options)
}
