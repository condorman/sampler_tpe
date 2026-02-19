import { MT19937 } from './mt19937.js'
import { randomSampleFromDistribution } from './randomSampling.js'

export class RandomSampler {
  constructor(seed = null) {
    this.rng = new MT19937(seed)
  }

  reseedRng() {
    this.rng.seed((Date.now() >>> 0) ^ 0x9e3779b9)
  }

  inferRelativeSearchSpace() {
    return {}
  }

  sampleRelative() {
    return {}
  }

  sampleIndependent(_study, _trial, _paramName, distribution) {
    return randomSampleFromDistribution(this.rng, distribution)
  }

  beforeTrial() {}

  afterTrial() {}
}
