export class FloatDistribution {
  constructor(low, high, log = false, step = null) {
    if (log && step !== null) {
      throw new Error('FloatDistribution: step is not supported when log is true.')
    }
    if (low > high) {
      throw new Error(`FloatDistribution: low <= high required, got low=${low}, high=${high}`)
    }
    if (log && low <= 0) {
      throw new Error(`FloatDistribution: low > 0 required for log=true, got low=${low}`)
    }
    if (step !== null && step <= 0) {
      throw new Error(`FloatDistribution: step > 0 required, got step=${step}`)
    }

    this.step = null
    if (step !== null) {
      this.step = Number(step)
      high = adjustDiscreteUniformHigh(low, high, step)
    }

    this.low = Number(low)
    this.high = Number(high)
    this.log = !!log
  }

  single() {
    if (this.step === null) {
      return this.low === this.high
    }
    if (this.low === this.high) {
      return true
    }
    return this.high - this.low < this.step
  }

  toInternalRepr(value) {
    const v = Number(value)
    if (Number.isNaN(v)) {
      throw new Error(`Invalid float value ${value}`)
    }
    if (this.log && v <= 0) {
      throw new Error(`Invalid value ${value} for log float distribution.`)
    }
    return v
  }

  toExternalRepr(value) {
    return value
  }

  equals(other) {
    return (
      other instanceof FloatDistribution &&
      this.low === other.low &&
      this.high === other.high &&
      this.log === other.log &&
      this.step === other.step
    )
  }
}

export class IntDistribution {
  constructor(low, high, log = false, step = 1) {
    if (log && step !== 1) {
      throw new Error('IntDistribution: step must be 1 when log=true.')
    }
    if (low > high) {
      throw new Error(`IntDistribution: low <= high required, got low=${low}, high=${high}`)
    }
    if (log && low < 1) {
      throw new Error(`IntDistribution: low >= 1 required for log=true, got low=${low}`)
    }
    if (step <= 0) {
      throw new Error(`IntDistribution: step > 0 required, got step=${step}`)
    }

    this.log = !!log
    this.step = Number(step)
    this.low = Number(low)
    this.high = Number(adjustIntUniformHigh(Number(low), Number(high), Number(step)))
  }

  toInternalRepr(value) {
    const v = Number(value)
    if (Number.isNaN(v)) {
      throw new Error(`Invalid int value ${value}`)
    }
    if (this.log && v <= 0) {
      throw new Error(`Invalid value ${value} for log int distribution.`)
    }
    return v
  }

  toExternalRepr(value) {
    return Math.trunc(value)
  }

  single() {
    if (this.log) {
      return this.low === this.high
    }
    if (this.low === this.high) {
      return true
    }
    return this.high - this.low < this.step
  }

  equals(other) {
    return (
      other instanceof IntDistribution &&
      this.low === other.low &&
      this.high === other.high &&
      this.log === other.log &&
      this.step === other.step
    )
  }
}

export class CategoricalDistribution {
  constructor(choices) {
    if (!Array.isArray(choices) || choices.length === 0) {
      throw new Error('CategoricalDistribution: choices must contain one or more elements.')
    }
    this.choices = [...choices]
  }

  toInternalRepr(value) {
    for (let i = 0; i < this.choices.length; i += 1) {
      if (Object.is(this.choices[i], value) || this.choices[i] === value) {
        return i
      }
    }
    throw new Error(`'${value}' not in categorical choices.`)
  }

  toExternalRepr(value) {
    return this.choices[Math.trunc(value)]
  }

  single() {
    return this.choices.length === 1
  }

  equals(other) {
    if (!(other instanceof CategoricalDistribution)) {
      return false
    }
    if (this.choices.length !== other.choices.length) {
      return false
    }
    for (let i = 0; i < this.choices.length; i += 1) {
      if (!Object.is(this.choices[i], other.choices[i]) && this.choices[i] !== other.choices[i]) {
        return false
      }
    }
    return true
  }
}

export function distributionContainsValue(distribution, value) {
  if (distribution instanceof CategoricalDistribution) {
    try {
      distribution.toInternalRepr(value)
      return true
    } catch {
      return false
    }
  }

  let numeric
  try {
    numeric = distribution.toInternalRepr(value)
  } catch {
    return false
  }
  if (!Number.isFinite(numeric)) {
    return false
  }

  const boundsTol = 1e-12
  if (distribution instanceof FloatDistribution) {
    if (numeric < distribution.low - boundsTol || numeric > distribution.high + boundsTol) {
      return false
    }
    if (distribution.step === null) {
      return true
    }
    const steps = (numeric - distribution.low) / distribution.step
    return Math.abs(steps - Math.round(steps)) <= 1e-10
  }

  if (distribution instanceof IntDistribution) {
    if (numeric < distribution.low - boundsTol || numeric > distribution.high + boundsTol) {
      return false
    }
    if (Math.abs(numeric - Math.round(numeric)) > 1e-10) {
      return false
    }
    const steps = (Math.round(numeric) - distribution.low) / distribution.step
    return Math.abs(steps - Math.round(steps)) <= 1e-10
  }

  return false
}

function adjustDiscreteUniformHigh(low, high, step) {
  const r = high - low
  const q = Math.floor(r / step)
  const adjusted = q * step + low
  if (Math.abs(adjusted - high) < 1e-15) {
    return high
  }
  return adjusted
}

function adjustIntUniformHigh(low, high, step) {
  const r = high - low
  const q = Math.floor(r / step)
  return q * step + low
}
