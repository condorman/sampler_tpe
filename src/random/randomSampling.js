import {
  CategoricalDistribution,
  FloatDistribution,
  IntDistribution
} from '../distributions/distributions.js'
import { transformNumericalParam, untransformNumericalParam } from '../distributions/numericTransform.js'

export function adjustDiscreteUniformHigh(low, high, step) {
  const r = high - low
  const q = Math.floor(r / step)
  const adjusted = q * step + low
  if (Math.abs(adjusted - high) < 1e-15) {
    return high
  }
  return adjusted
}

export function adjustIntUniformHigh(low, high, step) {
  const r = high - low
  const q = Math.floor(r / step)
  return q * step + low
}

export function randomSampleFromDistribution(rng, distribution) {
  if (distribution instanceof CategoricalDistribution) {
    let bestIdx = 0
    let bestValue = rng.uniform(0, 1)
    for (let i = 1; i < distribution.choices.length; i += 1) {
      const v = rng.uniform(0, 1)
      if (v > bestValue) {
        bestValue = v
        bestIdx = i
      }
    }
    return distribution.toExternalRepr(bestIdx)
  }

  if (distribution instanceof FloatDistribution) {
    let low
    let high
    if (distribution.step !== null) {
      const half = 0.5 * distribution.step
      low = transformNumericalParam(distribution.low, distribution, true) - half
      high = transformNumericalParam(distribution.high, distribution, true) + half
    } else {
      low = transformNumericalParam(distribution.low, distribution, true)
      high = transformNumericalParam(distribution.high, distribution, true)
    }
    const trans = rng.uniform(low, high)
    return untransformNumericalParam(trans, distribution, true)
  }

  if (distribution instanceof IntDistribution) {
    const half = 0.5 * distribution.step
    let low
    let high
    if (distribution.log) {
      low = transformNumericalParam(distribution.low - half, distribution, true)
      high = transformNumericalParam(distribution.high + half, distribution, true)
    } else {
      low = transformNumericalParam(distribution.low, distribution, true) - half
      high = transformNumericalParam(distribution.high, distribution, true) + half
    }
    const trans = rng.uniform(low, high)
    return untransformNumericalParam(trans, distribution, true)
  }

  throw new Error('Unsupported distribution in random sampler.')
}
