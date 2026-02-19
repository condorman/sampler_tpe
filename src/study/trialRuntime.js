import { FIXED_PARAMS_KEY } from '../core/constants.js'
import { hasOwn, isPlainObject } from '../core/objectUtils.js'
import {
  CategoricalDistribution,
  FloatDistribution,
  IntDistribution,
  distributionContainsValue
} from '../distributions/distributions.js'

export class TrialRuntime {
  constructor(study, frozenTrial) {
    this.study = study
    this.frozen = frozenTrial
    this.relativePrepared = false
    this.relativeSearchSpace = {}
    this.relativeParams = {}
  }

  get number() {
    return this.frozen.number
  }

  ensureRelativePrepared() {
    if (this.relativePrepared) {
      return
    }

    const sampler = this.study.sampler
    this.relativeSearchSpace = sampler.inferRelativeSearchSpace(this.study, this.frozen)
    this.relativeParams = sampler.sampleRelative(
      this.study,
      this.frozen,
      this.relativeSearchSpace
    )
    this.relativePrepared = true
  }

  _suggest(name, distribution) {
    if (name in this.frozen.params) {
      return this.frozen.params[name]
    }

    this.frozen.distributions[name] = distribution

    const fixedParams = this.frozen.system_attrs[FIXED_PARAMS_KEY]
    if (isPlainObject(fixedParams) && hasOwn(fixedParams, name)) {
      const fixedValue = fixedParams[name]
      if (!distributionContainsValue(distribution, fixedValue)) {
        console.warn(
          `Fixed parameter "${name}" with value ${String(fixedValue)} is out of range for the current distribution.`
        )
      }
      this.frozen.params[name] = fixedValue
      return fixedValue
    }

    this.ensureRelativePrepared()

    let value
    if (name in this.relativeSearchSpace && name in this.relativeParams) {
      value = this.relativeParams[name]
    } else {
      value = this.study.sampler.sampleIndependent(
        this.study,
        this.frozen,
        name,
        distribution
      )
    }

    this.frozen.params[name] = value
    return value
  }

  suggestFloat(name, low, high, options = {}) {
    const dist = new FloatDistribution(low, high, !!options.log, options.step ?? null)
    return this._suggest(name, dist)
  }

  suggestInt(name, low, high, options = {}) {
    const dist = new IntDistribution(low, high, !!options.log, options.step ?? 1)
    return this._suggest(name, dist)
  }

  suggestCategorical(name, choices) {
    const dist = new CategoricalDistribution(choices)
    return this._suggest(name, dist)
  }

  report(value, step) {
    this.frozen.intermediate_values[String(step)] = value
  }
}
