import { clip, nextafter, roundToNearestEven } from '../core/numberUtils.js'
import { FloatDistribution, IntDistribution } from './distributions.js'

export function transformNumericalParam(param, distribution, transformLog) {
  if (distribution instanceof FloatDistribution) {
    if (distribution.log) {
      return transformLog ? Math.log(param) : Number(param)
    }
    return Number(param)
  }
  if (distribution instanceof IntDistribution) {
    if (distribution.log) {
      return transformLog ? Math.log(param) : Number(param)
    }
    return Number(param)
  }
  throw new Error('Unexpected numerical distribution.')
}

export function untransformNumericalParam(transParam, distribution, transformLog) {
  if (distribution instanceof FloatDistribution) {
    if (distribution.log) {
      let param = transformLog ? Math.exp(transParam) : transParam
      if (!distribution.single()) {
        param = Math.min(param, nextafter(distribution.high, distribution.high - 1))
      }
      return param
    }
    if (distribution.step !== null) {
      return clip(
        roundToNearestEven((transParam - distribution.low) / distribution.step) * distribution.step +
          distribution.low,
        distribution.low,
        distribution.high
      )
    }
    if (distribution.single()) {
      return transParam
    }
    return Math.min(transParam, nextafter(distribution.high, distribution.high - 1))
  }

  if (distribution instanceof IntDistribution) {
    if (distribution.log) {
      if (transformLog) {
        return Math.trunc(
          clip(roundToNearestEven(Math.exp(transParam)), distribution.low, distribution.high)
        )
      }
      return Math.trunc(transParam)
    }
    return Math.trunc(
      clip(
        roundToNearestEven((transParam - distribution.low) / distribution.step) * distribution.step +
          distribution.low,
        distribution.low,
        distribution.high
      )
    )
  }

  throw new Error('Unexpected numerical distribution.')
}
