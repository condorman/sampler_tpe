import { EPS } from '../core/constants.js'
import { clip } from '../core/numberUtils.js'
import {
  CategoricalDistribution,
  FloatDistribution,
  IntDistribution
} from '../distributions/distributions.js'
import { numpyQuickArgSort } from '../math/sorting.js'
import { MixtureOfProductDistribution } from './mixtureOfProductDistribution.js'

export function defaultGamma(x) {
  return Math.min(Math.ceil(0.1 * x), 25)
}

export function defaultWeights(x) {
  if (x === 0) {
    return []
  }
  if (x < 25) {
    return new Array(x).fill(1)
  }
  const ramp = []
  const nRamp = x - 25
  if (nRamp === 1) {
    ramp.push(1 / x)
  } else {
    for (let i = 0; i < nRamp; i += 1) {
      ramp.push((1 / x) + ((1 - 1 / x) * i) / (nRamp - 1))
    }
  }
  const flat = new Array(25).fill(1)
  return [...ramp, ...flat]
}

export function callWeightsFunc(weightsFunc, n) {
  const w = weightsFunc(n).slice(0, n)
  for (const val of w) {
    if (val < 0) {
      throw new Error(`Weights function cannot return negative values: ${w}`)
    }
    if (!Number.isFinite(val)) {
      throw new Error(`Weights function cannot return inf or NaN values: ${w}`)
    }
  }
  if (w.length > 0) {
    let sum = 0
    for (const val of w) sum += val
    if (sum <= 0) {
      throw new Error(`Weights function cannot return all-zero values: ${w}`)
    }
  }
  return w
}

export class ParzenEstimator {
  constructor(observations, searchSpace, parameters, predeterminedWeights = null) {
    if (parameters.priorWeight < 0) {
      throw new Error(`priorWeight must be non-negative, got ${parameters.priorWeight}`)
    }

    this.searchSpace = searchSpace

    const transformed = this.transform(observations)
    if (predeterminedWeights !== null && transformed.length !== predeterminedWeights.length) {
      throw new Error('predetermined weights length mismatch.')
    }

    let weights
    if (predeterminedWeights !== null) {
      weights = [...predeterminedWeights]
    } else {
      weights = callWeightsFunc(parameters.weights, transformed.length)
    }

    if (transformed.length === 0) {
      weights = [1]
    } else {
      weights = [...weights, parameters.priorWeight]
    }

    let sum = 0
    for (const w of weights) sum += w
    weights = weights.map((w) => w / sum)

    const distributions = []
    const paramNames = Object.keys(searchSpace)
    for (let i = 0; i < paramNames.length; i += 1) {
      const param = paramNames[i]
      const obsForParam = transformed.map((row) => row[i])
      distributions.push({
        paramName: param,
        distribution: this.calculateDistribution(obsForParam, param, searchSpace[param], parameters)
      })
    }

    this.mixture = new MixtureOfProductDistribution(weights, distributions)
  }

  transform(samplesByParam) {
    const paramNames = Object.keys(this.searchSpace)
    if (paramNames.length === 0) {
      return []
    }

    const n = samplesByParam[paramNames[0]] ? samplesByParam[paramNames[0]].length : 0
    const rows = new Array(n)
    for (let i = 0; i < n; i += 1) {
      rows[i] = paramNames.map((param) => samplesByParam[param][i])
    }
    return rows
  }

  calculateDistribution(observations, paramName, searchSpace, parameters) {
    if (searchSpace instanceof CategoricalDistribution) {
      return this.calculateCategoricalDistributions(observations, paramName, searchSpace, parameters)
    }
    if (searchSpace instanceof FloatDistribution || searchSpace instanceof IntDistribution) {
      return this.calculateNumericalDistributions(observations, searchSpace, parameters)
    }
    throw new Error('Unsupported distribution in parzen estimator.')
  }

  calculateCategoricalDistributions(observations, paramName, searchSpace, parameters) {
    const choices = searchSpace.choices
    const nChoices = choices.length

    if (observations.length === 0) {
      return {
        kind: 'categorical',
        weights: [[...new Array(nChoices).fill(1 / nChoices)]]
      }
    }

    const nKernels = observations.length + 1
    const weights = new Array(nKernels)
    for (let k = 0; k < nKernels; k += 1) {
      weights[k] = new Array(nChoices).fill(parameters.priorWeight / nKernels)
    }

    for (let i = 0; i < observations.length; i += 1) {
      const idx = Math.trunc(observations[i])
      weights[i][idx] += 1
    }

    for (let i = 0; i < nKernels; i += 1) {
      let rowSum = 0
      for (let j = 0; j < nChoices; j += 1) rowSum += weights[i][j]
      if (rowSum === 0) {
        continue
      }
      for (let j = 0; j < nChoices; j += 1) {
        weights[i][j] /= rowSum
      }
    }

    return {
      kind: 'categorical',
      weights
    }
  }

  calculateNumericalDistributions(observationsInput, searchSpace, parameters) {
    const observations = [...observationsInput]

    let low = searchSpace.low
    let high = searchSpace.high
    if (searchSpace.step !== null && searchSpace.step !== undefined) {
      low -= searchSpace.step / 2
      high += searchSpace.step / 2
    }

    if (searchSpace.log) {
      for (let i = 0; i < observations.length; i += 1) {
        observations[i] = Math.log(observations[i])
      }
      low = Math.log(low)
      high = Math.log(high)
    }

    let mus = [...observations]

    const computeSigmas = () => {
      let sigmas
      if (parameters.multivariate) {
        const sigma0Magnitude = 0.2
        const sigma = sigma0Magnitude *
          Math.pow(Math.max(observations.length, 1), -1 / (Object.keys(this.searchSpace).length + 4)) *
          (high - low)
        sigmas = new Array(observations.length).fill(sigma)
      } else {
        const priorMu = 0.5 * (low + high)
        const musWithPrior = [...mus, priorMu]
        const sortedIndices = numpyQuickArgSort(musWithPrior)
        const sortedMus = sortedIndices.map((i) => musWithPrior[i])

        const sortedMusWithEndpoints = [low, ...sortedMus, high]
        const sortedSigmas = new Array(sortedMus.length)
        for (let i = 0; i < sortedMus.length; i += 1) {
          const left = sortedMusWithEndpoints[i + 1] - sortedMusWithEndpoints[i]
          const right = sortedMusWithEndpoints[i + 2] - sortedMusWithEndpoints[i + 1]
          sortedSigmas[i] = Math.max(left, right)
        }

        if (!parameters.considerEndpoints && sortedMusWithEndpoints.length >= 4) {
          sortedSigmas[0] = sortedMusWithEndpoints[2] - sortedMusWithEndpoints[1]
          sortedSigmas[sortedSigmas.length - 1] =
            sortedMusWithEndpoints[sortedMusWithEndpoints.length - 2] -
            sortedMusWithEndpoints[sortedMusWithEndpoints.length - 3]
        }

        const inverse = new Array(sortedIndices.length)
        for (let i = 0; i < sortedIndices.length; i += 1) {
          inverse[sortedIndices[i]] = i
        }

        sigmas = new Array(observations.length)
        for (let i = 0; i < observations.length; i += 1) {
          sigmas[i] = sortedSigmas[inverse[i]]
        }
      }

      const maxSigma = high - low
      let minSigma
      if (parameters.considerMagicClip) {
        const nKernels = observations.length + 1
        minSigma = (high - low) / Math.min(100, 1 + nKernels)
      } else {
        minSigma = EPS
      }

      return sigmas.map((s) => clip(s, minSigma, maxSigma))
    }

    let sigmas = computeSigmas()
    mus = [...mus, 0.5 * (low + high)]
    sigmas = [...sigmas, high - low]

    if (searchSpace.step === null || searchSpace.step === undefined) {
      if (!searchSpace.log) {
        return {
          kind: 'truncnorm',
          mu: mus,
          sigma: sigmas,
          low: searchSpace.low,
          high: searchSpace.high
        }
      }
      return {
        kind: 'trunclognorm',
        mu: mus,
        sigma: sigmas,
        low: searchSpace.low,
        high: searchSpace.high
      }
    }

    if (!searchSpace.log) {
      return {
        kind: 'discrete_truncnorm',
        mu: mus,
        sigma: sigmas,
        low: searchSpace.low,
        high: searchSpace.high,
        step: searchSpace.step
      }
    }
    return {
      kind: 'discrete_trunclognorm',
      mu: mus,
      sigma: sigmas,
      low: searchSpace.low,
      high: searchSpace.high,
      step: searchSpace.step
    }
  }

  sample(rng, size) {
    return this.mixture.sample(rng, size)
  }

  logPdf(samples) {
    return this.mixture.logPdf(samples)
  }
}
