const EPS = 1e-12
const CONSTRAINTS_KEY = 'constraints'

export const TrialState = {
  RUNNING: 'running',
  COMPLETE: 'complete',
  PRUNED: 'pruned',
  FAIL: 'fail',
  WAITING: 'waiting'
}

export const StudyDirection = {
  MINIMIZE: 'minimize',
  MAXIMIZE: 'maximize'
}

const STUDY_SNAPSHOT_MAGIC = 'optuna_tpe_study_snapshot'
const STUDY_SNAPSHOT_VERSION = 1
const SPECIAL_NUMBER_MARKER = '__optuna_tpe_special_number__'
const FUNCTION_SPEC_KIND = {
  BUILTIN: 'builtin',
  CUSTOM: 'custom',
  NONE: 'none'
}

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function encodeNumberForSnapshot(value) {
  if (Number.isNaN(value)) {
    return { [SPECIAL_NUMBER_MARKER]: 'NaN' }
  }
  if (value === Infinity) {
    return { [SPECIAL_NUMBER_MARKER]: '+Infinity' }
  }
  if (value === -Infinity) {
    return { [SPECIAL_NUMBER_MARKER]: '-Infinity' }
  }
  if (Object.is(value, -0)) {
    return { [SPECIAL_NUMBER_MARKER]: '-0' }
  }
  return value
}

function decodeNumberFromSnapshot(value) {
  if (
    !isPlainObject(value) ||
    !(SPECIAL_NUMBER_MARKER in value) ||
    Object.keys(value).length !== 1
  ) {
    return value
  }
  const token = value[SPECIAL_NUMBER_MARKER]
  if (token === 'NaN') return Number.NaN
  if (token === '+Infinity') return Infinity
  if (token === '-Infinity') return -Infinity
  if (token === '-0') return -0
  throw new Error(`Unknown serialized number token: ${token}`)
}

function serializeJsonValueForSnapshot(value) {
  if (value === null) {
    return null
  }

  const valueType = typeof value
  if (valueType === 'number') {
    return encodeNumberForSnapshot(value)
  }
  if (valueType === 'string' || valueType === 'boolean') {
    return value
  }
  if (Array.isArray(value)) {
    return value.map((item) => serializeJsonValueForSnapshot(item))
  }
  if (isPlainObject(value)) {
    const out = {}
    for (const [key, item] of Object.entries(value)) {
      out[key] = serializeJsonValueForSnapshot(item)
    }
    return out
  }

  throw new Error(`Cannot serialize value of type "${valueType}" in study snapshot.`)
}

function deserializeJsonValueFromSnapshot(value) {
  if (value === null) {
    return null
  }
  if (typeof value !== 'object') {
    return value
  }
  if (Array.isArray(value)) {
    return value.map((item) => deserializeJsonValueFromSnapshot(item))
  }
  if (SPECIAL_NUMBER_MARKER in value && Object.keys(value).length === 1) {
    return decodeNumberFromSnapshot(value)
  }
  const out = {}
  for (const [key, item] of Object.entries(value)) {
    out[key] = deserializeJsonValueFromSnapshot(item)
  }
  return out
}

function serializeDistributionForSnapshot(distribution) {
  if (distribution instanceof FloatDistribution) {
    return {
      type: 'FloatDistribution',
      low: serializeJsonValueForSnapshot(distribution.low),
      high: serializeJsonValueForSnapshot(distribution.high),
      log: distribution.log,
      step: distribution.step === null ? null : serializeJsonValueForSnapshot(distribution.step)
    }
  }
  if (distribution instanceof IntDistribution) {
    return {
      type: 'IntDistribution',
      low: serializeJsonValueForSnapshot(distribution.low),
      high: serializeJsonValueForSnapshot(distribution.high),
      log: distribution.log,
      step: serializeJsonValueForSnapshot(distribution.step)
    }
  }
  if (distribution instanceof CategoricalDistribution) {
    return {
      type: 'CategoricalDistribution',
      choices: distribution.choices.map((choice) => serializeJsonValueForSnapshot(choice))
    }
  }
  throw new Error('Unsupported distribution found while serializing study snapshot.')
}

function deserializeDistributionFromSnapshot(payload) {
  if (!isPlainObject(payload) || typeof payload.type !== 'string') {
    throw new Error('Invalid serialized distribution payload.')
  }

  if (payload.type === 'FloatDistribution') {
    return new FloatDistribution(
      deserializeJsonValueFromSnapshot(payload.low),
      deserializeJsonValueFromSnapshot(payload.high),
      !!payload.log,
      payload.step === null ? null : deserializeJsonValueFromSnapshot(payload.step)
    )
  }
  if (payload.type === 'IntDistribution') {
    return new IntDistribution(
      deserializeJsonValueFromSnapshot(payload.low),
      deserializeJsonValueFromSnapshot(payload.high),
      !!payload.log,
      deserializeJsonValueFromSnapshot(payload.step)
    )
  }
  if (payload.type === 'CategoricalDistribution') {
    const choices = (payload.choices || []).map((choice) => deserializeJsonValueFromSnapshot(choice))
    return new CategoricalDistribution(choices)
  }

  throw new Error(`Unknown serialized distribution type "${payload.type}".`)
}

function serializeRngStateForSnapshot(rng) {
  return {
    mt: Array.from(rng.mt),
    mti: rng.mti
  }
}

function restoreRngStateFromSnapshot(rng, state) {
  if (!isPlainObject(state) || !Array.isArray(state.mt) || typeof state.mti !== 'number') {
    throw new Error('Invalid RNG state payload in study snapshot.')
  }
  if (state.mt.length !== rng.N) {
    throw new Error(`Invalid RNG state length: expected ${rng.N}, got ${state.mt.length}.`)
  }
  for (let i = 0; i < rng.N; i += 1) {
    const value = Number(state.mt[i])
    if (!Number.isFinite(value)) {
      throw new Error('RNG state contains non-finite values.')
    }
    rng.mt[i] = value >>> 0
  }
  rng.mti = Number(state.mti)
}

function serializeFunctionSpec(fn, builtinFn, builtinName) {
  if (fn === builtinFn) {
    return {
      kind: FUNCTION_SPEC_KIND.BUILTIN,
      name: builtinName
    }
  }
  return {
    kind: FUNCTION_SPEC_KIND.CUSTOM,
    name: typeof fn === 'function' && fn.name ? fn.name : null
  }
}

function resolveFunctionSpec(spec, builtinFn, builtinName, overrideFn, label) {
  if (!isPlainObject(spec) || typeof spec.kind !== 'string') {
    throw new Error(`Invalid serialized function spec for "${label}".`)
  }
  if (spec.kind === FUNCTION_SPEC_KIND.BUILTIN) {
    if (spec.name !== builtinName) {
      throw new Error(`Unknown builtin function "${spec.name}" for "${label}".`)
    }
    return builtinFn
  }
  if (spec.kind === FUNCTION_SPEC_KIND.CUSTOM) {
    if (typeof overrideFn !== 'function') {
      throw new Error(
        `Serialized study requires custom function for "${label}". Provide it via Study.deserialize(..., { samplerFunctions: { ${label}: fn } }).`
      )
    }
    return overrideFn
  }
  throw new Error(`Unsupported function spec kind "${spec.kind}" for "${label}".`)
}

function serializeOptionalFunctionSpec(fn) {
  if (fn === null || fn === undefined) {
    return { kind: FUNCTION_SPEC_KIND.NONE }
  }
  return {
    kind: FUNCTION_SPEC_KIND.CUSTOM,
    name: typeof fn === 'function' && fn.name ? fn.name : null
  }
}

function resolveOptionalFunctionSpec(spec, overrideFn, label) {
  if (!isPlainObject(spec) || typeof spec.kind !== 'string') {
    throw new Error(`Invalid optional function spec for "${label}".`)
  }
  if (spec.kind === FUNCTION_SPEC_KIND.NONE) {
    return null
  }
  if (spec.kind === FUNCTION_SPEC_KIND.CUSTOM) {
    if (typeof overrideFn !== 'function') {
      throw new Error(
        `Serialized study requires custom function for "${label}". Provide it via Study.deserialize(..., { samplerFunctions: { ${label}: fn } }).`
      )
    }
    return overrideFn
  }
  throw new Error(`Unsupported optional function spec kind "${spec.kind}" for "${label}".`)
}

function serializeCategoricalDistanceSpec(distanceFuncMap) {
  if (!distanceFuncMap || Object.keys(distanceFuncMap).length === 0) {
    return { kind: FUNCTION_SPEC_KIND.NONE }
  }
  const keys = Object.keys(distanceFuncMap).sort()
  return {
    kind: FUNCTION_SPEC_KIND.CUSTOM,
    keys
  }
}

function resolveCategoricalDistanceSpec(spec, overrideMap) {
  if (!isPlainObject(spec) || typeof spec.kind !== 'string') {
    throw new Error('Invalid categorical distance function spec in snapshot.')
  }
  if (spec.kind === FUNCTION_SPEC_KIND.NONE) {
    return null
  }
  if (spec.kind === FUNCTION_SPEC_KIND.CUSTOM) {
    if (!isPlainObject(overrideMap)) {
      throw new Error(
        'Serialized study requires categoricalDistanceFunc map. Provide it via Study.deserialize(..., { samplerFunctions: { categoricalDistanceFunc: { ... } } }).'
      )
    }
    for (const key of spec.keys || []) {
      if (typeof overrideMap[key] !== 'function') {
        throw new Error(`Missing categorical distance function for key "${key}".`)
      }
    }
    return overrideMap
  }
  throw new Error(`Unsupported categorical distance function spec kind "${spec.kind}".`)
}

function serializeSamplerForSnapshot(sampler) {
  if (!(sampler instanceof TPESampler)) {
    throw new Error('Only TPESampler serialization is currently supported.')
  }
  return {
    samplerType: 'TPESampler',
    config: {
      priorWeight: sampler.parzenEstimatorParameters.priorWeight,
      considerMagicClip: sampler.parzenEstimatorParameters.considerMagicClip,
      considerEndpoints: sampler.parzenEstimatorParameters.considerEndpoints,
      nStartupTrials: sampler.nStartupTrials,
      nEiCandidates: sampler.nEiCandidates,
      multivariate: sampler.multivariate,
      group: sampler.group,
      warnIndependentSampling: sampler.warnIndependentSampling,
      constantLiar: sampler.constantLiar,
      gamma: serializeFunctionSpec(sampler.gamma, defaultGamma, 'defaultGamma'),
      weights: serializeFunctionSpec(
        sampler.parzenEstimatorParameters.weights,
        defaultWeights,
        'defaultWeights'
      ),
      constraintsFunc: serializeOptionalFunctionSpec(sampler.constraintsFunc),
      categoricalDistanceFunc: serializeCategoricalDistanceSpec(
        sampler.parzenEstimatorParameters.categoricalDistanceFunc
      )
    },
    rngState: serializeRngStateForSnapshot(sampler.rng),
    randomSamplerRngState: serializeRngStateForSnapshot(sampler.randomSampler.rng)
  }
}

function deserializeSamplerFromSnapshot(payload, options = {}) {
  if (!isPlainObject(payload) || payload.samplerType !== 'TPESampler') {
    throw new Error('Unsupported sampler snapshot payload.')
  }
  const functions = (options && options.samplerFunctions) || {}
  const config = payload.config || {}
  const sampler = new TPESampler({
    priorWeight: config.priorWeight,
    considerMagicClip: !!config.considerMagicClip,
    considerEndpoints: !!config.considerEndpoints,
    nStartupTrials: config.nStartupTrials,
    nEiCandidates: config.nEiCandidates,
    gamma: resolveFunctionSpec(config.gamma, defaultGamma, 'defaultGamma', functions.gamma, 'gamma'),
    weights: resolveFunctionSpec(
      config.weights,
      defaultWeights,
      'defaultWeights',
      functions.weights,
      'weights'
    ),
    seed: 0,
    multivariate: !!config.multivariate,
    group: !!config.group,
    warnIndependentSampling: !!config.warnIndependentSampling,
    constantLiar: !!config.constantLiar,
    constraintsFunc: resolveOptionalFunctionSpec(
      config.constraintsFunc,
      functions.constraintsFunc,
      'constraintsFunc'
    ),
    categoricalDistanceFunc: resolveCategoricalDistanceSpec(
      config.categoricalDistanceFunc,
      functions.categoricalDistanceFunc
    )
  })

  restoreRngStateFromSnapshot(sampler.rng, payload.rngState)
  restoreRngStateFromSnapshot(sampler.randomSampler.rng, payload.randomSamplerRngState)
  return sampler
}

function serializeTrialForSnapshot(trial) {
  const distributions = {}
  for (const [name, distribution] of Object.entries(trial.distributions || {})) {
    distributions[name] = serializeDistributionForSnapshot(distribution)
  }
  return {
    number: trial.number,
    state: trial.state,
    params: serializeJsonValueForSnapshot(trial.params || {}),
    distributions,
    system_attrs: serializeJsonValueForSnapshot(trial.system_attrs || {}),
    intermediate_values: serializeJsonValueForSnapshot(trial.intermediate_values || {}),
    value: serializeJsonValueForSnapshot(trial.value),
    values: serializeJsonValueForSnapshot(trial.values)
  }
}

function deserializeTrialFromSnapshot(payload) {
  if (!isPlainObject(payload)) {
    throw new Error('Invalid trial payload in serialized study.')
  }
  const distributions = {}
  for (const [name, distributionPayload] of Object.entries(payload.distributions || {})) {
    distributions[name] = deserializeDistributionFromSnapshot(distributionPayload)
  }
  return {
    number: payload.number,
    state: payload.state,
    params: deserializeJsonValueFromSnapshot(payload.params || {}),
    distributions,
    system_attrs: deserializeJsonValueFromSnapshot(payload.system_attrs || {}),
    intermediate_values: deserializeJsonValueFromSnapshot(payload.intermediate_values || {}),
    value: deserializeJsonValueFromSnapshot(payload.value),
    values: deserializeJsonValueFromSnapshot(payload.values)
  }
}

function isFinishedState(state) {
  return state !== TrialState.RUNNING && state !== TrialState.WAITING
}

function clip(value, low, high) {
  if (value < low) return low
  if (value > high) return high
  return value
}

function logsumexp(values) {
  let maxValue = -Infinity
  for (let i = 0; i < values.length; i += 1) {
    if (values[i] > maxValue) maxValue = values[i]
  }
  if (!Number.isFinite(maxValue)) {
    return maxValue
  }
  let acc = 0
  for (let i = 0; i < values.length; i += 1) {
    acc += Math.exp(values[i] - maxValue)
  }
  return Math.log(acc) + maxValue
}

function roundToNearestEven(value) {
  const floorValue = Math.floor(value)
  const fraction = value - floorValue
  const eps = 1e-12
  if (fraction < 0.5 - eps) {
    return floorValue
  }
  if (fraction > 0.5 + eps) {
    return floorValue + 1
  }
  return floorValue % 2 === 0 ? floorValue : floorValue + 1
}

function nextafter(x, y) {
  if (Number.isNaN(x) || Number.isNaN(y)) return Number.NaN
  if (x === y) return y
  if (!Number.isFinite(x)) return x
  if (x === 0) {
    return y > 0 ? Number.MIN_VALUE : -Number.MIN_VALUE
  }

  const buffer = new ArrayBuffer(8)
  const view = new DataView(buffer)
  view.setFloat64(0, x, false)
  const hi = BigInt(view.getUint32(0, false))
  const lo = BigInt(view.getUint32(4, false))
  let bits = (hi << 32n) | lo

  const increment = (y > x) === (x > 0)
  bits = increment ? bits + 1n : bits - 1n

  view.setUint32(0, Number((bits >> 32n) & 0xffffffffn), false)
  view.setUint32(4, Number(bits & 0xffffffffn), false)
  return view.getFloat64(0, false)
}

function shallowCopy(obj) {
  return Object.assign({}, obj)
}

function sortObjectEntries(obj) {
  return Object.keys(obj)
    .sort()
    .map((key) => [key, obj[key]])
}

function arraysEqual(a, b) {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false
  }
  return true
}

function compareRowsLex(a, b) {
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] < b[i]) return -1
    if (a[i] > b[i]) return 1
  }
  return 0
}

function uniqueSortedRowsWithInverse(rows) {
  const order = rows.map((_, i) => i).sort((ia, ib) => compareRowsLex(rows[ia], rows[ib]))
  const uniqueRows = []
  const inverse = new Array(rows.length)

  for (let i = 0; i < order.length; i += 1) {
    const idx = order[i]
    const row = rows[idx]
    if (uniqueRows.length === 0 || compareRowsLex(row, uniqueRows[uniqueRows.length - 1]) !== 0) {
      uniqueRows.push(row.slice())
    }
    inverse[idx] = uniqueRows.length - 1
  }

  return { uniqueRows, inverse }
}

function getMsb(n) {
  let msb = -1
  while (n > 0) {
    n >>= 1
    msb += 1
  }
  return msb
}

function heapSortArgSegment(values, indices, start, end) {
  const less = (i, j) => values[indices[i]] < values[indices[j]]
  const swap = (i, j) => {
    const tmp = indices[i]
    indices[i] = indices[j]
    indices[j] = tmp
  }
  const length = end - start + 1
  const siftDown = (root, size) => {
    while (true) {
      let child = root * 2 + 1
      if (child >= size) return
      if (child + 1 < size && less(start + child, start + child + 1)) {
        child += 1
      }
      if (!less(start + root, start + child)) {
        return
      }
      swap(start + root, start + child)
      root = child
    }
  }
  for (let i = Math.floor(length / 2) - 1; i >= 0; i -= 1) {
    siftDown(i, length)
  }
  for (let i = length - 1; i > 0; i -= 1) {
    swap(start, start + i)
    siftDown(0, i)
  }
}

function numpyQuickArgSort(values) {
  const num = values.length
  const tosort = Array.from({ length: num }, (_, i) => i)
  if (num <= 1) {
    return tosort
  }

  const SMALL_QUICKSORT = 15
  const stack = []
  let pl = 0
  let pr = num - 1
  let cdepth = getMsb(num) * 2

  const lessValue = (idxA, idxB) => values[tosort[idxA]] < values[tosort[idxB]]
  const lessPivot = (value, idxB) => value < values[tosort[idxB]]
  const lessFrom = (idxA, value) => values[tosort[idxA]] < value
  const swap = (a, b) => {
    const tmp = tosort[a]
    tosort[a] = tosort[b]
    tosort[b] = tmp
  }

  for (;;) {
    if (cdepth < 0) {
      heapSortArgSegment(values, tosort, pl, pr)
      if (stack.length === 0) {
        break
      }
      const last = stack.pop()
      pl = last.pl
      pr = last.pr
      cdepth = last.cdepth
      continue
    }

    while (pr - pl > SMALL_QUICKSORT) {
      let pm = pl + ((pr - pl) >> 1)
      if (lessValue(pm, pl)) {
        swap(pm, pl)
      }
      if (lessValue(pr, pm)) {
        swap(pr, pm)
      }
      if (lessValue(pm, pl)) {
        swap(pm, pl)
      }

      const vp = values[tosort[pm]]
      let pi = pl
      let pj = pr - 1
      swap(pm, pj)

      for (;;) {
        do {
          pi += 1
        } while (lessFrom(pi, vp))
        do {
          pj -= 1
        } while (lessPivot(vp, pj))
        if (pi >= pj) {
          break
        }
        swap(pi, pj)
      }

      swap(pi, pr - 1)

      cdepth -= 1
      if (pi - pl < pr - pi) {
        stack.push({ pl: pi + 1, pr, cdepth })
        pr = pi - 1
      } else {
        stack.push({ pl, pr: pi - 1, cdepth })
        pl = pi + 1
      }
    }

    for (let pi = pl + 1; pi <= pr; pi += 1) {
      const vi = tosort[pi]
      const vp = values[vi]
      let pj = pi
      let pk = pi - 1
      while (pj > pl && vp < values[tosort[pk]]) {
        tosort[pj] = tosort[pk]
        pj -= 1
        pk -= 1
      }
      tosort[pj] = vi
    }

    if (stack.length === 0) {
      break
    }

    const last = stack.pop()
    pl = last.pl
    pr = last.pr
    cdepth = last.cdepth
  }

  return tosort
}

function randomChoiceIndicesByWeights(rng, probabilities, size) {
  const cumulative = new Array(probabilities.length)
  let running = 0
  for (let i = 0; i < probabilities.length; i += 1) {
    running += probabilities[i]
    cumulative[i] = running
  }
  cumulative[cumulative.length - 1] = 1

  const out = new Array(size)
  for (let i = 0; i < size; i += 1) {
    const r = rng.randomSample()
    let idx = 0
    while (idx < cumulative.length && r >= cumulative[idx]) {
      idx += 1
    }
    out[i] = idx
  }
  return out
}

class MT19937 {
  constructor(seed = null) {
    this.N = 624
    this.M = 397
    this.MATRIX_A = 0x9908b0df
    this.UPPER_MASK = 0x80000000
    this.LOWER_MASK = 0x7fffffff
    this.mt = new Uint32Array(this.N)
    this.mti = this.N + 1
    if (seed !== null && seed !== undefined) {
      this.seed(seed)
    } else {
      this.seed((Date.now() >>> 0) ^ 0x6c078965)
    }
  }

  seed(seed) {
    let s = Number(seed) >>> 0
    this.mt[0] = s
    for (this.mti = 1; this.mti < this.N; this.mti += 1) {
      s = this.mt[this.mti - 1] ^ (this.mt[this.mti - 1] >>> 30)
      // keep in uint32 domain.
      this.mt[this.mti] = ((((s & 0xffff0000) >>> 16) * 1812433253) << 16) +
        (s & 0x0000ffff) * 1812433253 + this.mti
      this.mt[this.mti] >>>= 0
    }
  }

  _genInt32() {
    let y
    const mag01 = [0x0, this.MATRIX_A]

    if (this.mti >= this.N) {
      let kk
      for (kk = 0; kk < this.N - this.M; kk += 1) {
        y = (this.mt[kk] & this.UPPER_MASK) | (this.mt[kk + 1] & this.LOWER_MASK)
        this.mt[kk] = this.mt[kk + this.M] ^ (y >>> 1) ^ mag01[y & 0x1]
      }
      for (; kk < this.N - 1; kk += 1) {
        y = (this.mt[kk] & this.UPPER_MASK) | (this.mt[kk + 1] & this.LOWER_MASK)
        this.mt[kk] = this.mt[kk + (this.M - this.N)] ^ (y >>> 1) ^ mag01[y & 0x1]
      }
      y = (this.mt[this.N - 1] & this.UPPER_MASK) | (this.mt[0] & this.LOWER_MASK)
      this.mt[this.N - 1] = this.mt[this.M - 1] ^ (y >>> 1) ^ mag01[y & 0x1]
      this.mti = 0
    }

    y = this.mt[this.mti]
    this.mti += 1

    y ^= y >>> 11
    y ^= (y << 7) & 0x9d2c5680
    y ^= (y << 15) & 0xefc60000
    y ^= y >>> 18

    return y >>> 0
  }

  randomSample() {
    const a = this._genInt32() >>> 5
    const b = this._genInt32() >>> 6
    return (a * 67108864 + b) / 9007199254740992
  }

  rand(size) {
    const out = new Array(size)
    for (let i = 0; i < size; i += 1) {
      out[i] = this.randomSample()
    }
    return out
  }

  uniform(low, high) {
    return low + (high - low) * this.randomSample()
  }

  choiceWeighted(probabilities, size) {
    return randomChoiceIndicesByWeights(this, probabilities, size)
  }
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
    return (this.high - this.low) < this.step
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
    return (this.high - this.low) < this.step
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

function transformNumericalParam(param, distribution, transformLog) {
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

function untransformNumericalParam(transParam, distribution, transformLog) {
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

function randomSampleFromDistribution(rng, distribution) {
  if (distribution instanceof CategoricalDistribution) {
    // Match Optuna RandomSampler behavior via _SearchSpaceTransform one-hot argmax.
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

class RandomSampler {
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

class SearchSpaceGroup {
  constructor() {
    this.searchSpaces = []
  }

  addDistributions(distributions) {
    let distKeys = new Set(Object.keys(distributions))
    const nextSpaces = []

    for (const searchSpace of this.searchSpaces) {
      const keys = new Set(Object.keys(searchSpace))
      const intersect = {}
      const left = {}
      for (const key of keys) {
        if (distKeys.has(key)) {
          intersect[key] = searchSpace[key]
        } else {
          left[key] = searchSpace[key]
        }
      }
      if (Object.keys(intersect).length > 0) nextSpaces.push(intersect)
      if (Object.keys(left).length > 0) nextSpaces.push(left)
      for (const key of keys) {
        distKeys.delete(key)
      }
    }

    const right = {}
    for (const key of distKeys) {
      right[key] = distributions[key]
    }
    if (Object.keys(right).length > 0) nextSpaces.push(right)
    this.searchSpaces = nextSpaces
  }

  clone() {
    const c = new SearchSpaceGroup()
    c.searchSpaces = this.searchSpaces.map((sp) => shallowCopy(sp))
    return c
  }
}

class GroupDecomposedSearchSpace {
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

class IntersectionSearchSpace {
  constructor(includePruned = false) {
    this.includePruned = includePruned
    this.cachedTrialNumber = -1
    this.searchSpace = null
  }

  calculate(study, useCache = false) {
    const statesOfInterest = [TrialState.COMPLETE, TrialState.WAITING, TrialState.RUNNING]
    if (this.includePruned) {
      statesOfInterest.push(TrialState.PRUNED)
    }

    let nextCachedTrialNumber = -1
    const trials = study.getTrials({ useCache })
    for (let i = trials.length - 1; i >= 0; i -= 1) {
      const trial = trials[i]
      if (!statesOfInterest.includes(trial.state)) {
        continue
      }

      if (nextCachedTrialNumber === -1) {
        nextCachedTrialNumber = trial.number + 1
      }

      if (this.cachedTrialNumber > trial.number) {
        break
      }

      if (!isFinishedState(trial.state)) {
        nextCachedTrialNumber = trial.number
        continue
      }

      if (this.searchSpace === null) {
        this.searchSpace = shallowCopy(trial.distributions)
        continue
      }

      const next = {}
      for (const [name, distribution] of Object.entries(this.searchSpace)) {
        const trialDist = trial.distributions[name]
        if (trialDist && trialDist.equals(distribution)) {
          next[name] = distribution
        }
      }
      this.searchSpace = next
    }

    this.cachedTrialNumber = nextCachedTrialNumber
    const result = this.searchSpace || {}
    const sortedResult = {}
    for (const [name, distribution] of sortObjectEntries(result)) {
      sortedResult[name] = distribution
    }
    return sortedResult
  }
}

const SQRT2 = Math.SQRT2
const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI)
const NDTRI_EXP_APPROX_C = Math.sqrt(3) / Math.PI

function polyEval(coeffs, x) {
  let result = 0
  for (let i = coeffs.length - 1; i >= 0; i -= 1) {
    result = result * x + coeffs[i]
  }
  return result
}

const ERF_COEFF = {
  erx: 8.45062911510467529297e-01,
  efx: 1.28379167095512586316e-01,
  pp: [1.28379167095512558561e-01, -3.25042107247001499370e-01, -2.84817495755985104766e-02, -5.77027029648944159157e-03, -2.37630166566501626084e-05],
  qq: [1, 3.97917223959155352819e-01, 6.50222499887672944485e-02, 5.08130628187576562776e-03, 1.32494738004321644526e-04, -3.96022827877536812320e-06],
  pa: [-2.36211856075265944077e-03, 4.14856118683748331666e-01, -3.72207876035701323847e-01, 3.18346619901161753674e-01, -1.10894694282396677476e-01, 3.54783043256182359371e-02, -2.16637559486879084300e-03],
  qa: [1, 1.06420880400844228286e-01, 5.40397917702171048937e-01, 7.18286544141962662868e-02, 1.26171219808761642112e-01, 1.36370839120290507362e-02, 1.19844998467991074170e-02],
  ra: [-9.86494403484714822705e-03, -6.93858572707181764372e-01, -1.05586262253232909814e01, -6.23753324503260060396e01, -1.62396669462573470355e02, -1.84605092906711035994e02, -8.12874355063065934246e01, -9.81432934416914548592e00],
  sa: [1, 1.96512716674392571292e01, 1.37657754143519042600e02, 4.34565877475229228821e02, 6.45387271733267880336e02, 4.29008140027567833386e02, 1.08635005541779435134e02, 6.57024977031928170135e00, -6.04244152148580987438e-02],
  rb: [-9.86494292470009928597e-03, -7.99283237680523006574e-01, -1.77579549177547519889e01, -1.60636384855821916062e02, -6.37566443368389627722e02, -1.02509513161107724954e03, -4.83519191608651397019e02],
  sb: [1, 3.03380607434824582924e01, 3.25792512996573918826e02, 1.53672958608443695994e03, 3.19985821950859553908e03, 2.55305040643316442583e03, 4.74528541206955367215e02, -2.24409524465858183362e01]
}

function erfScalar(x) {
  if (Number.isNaN(x)) return Number.NaN
  const sign = x < 0 ? -1 : 1
  const a = Math.abs(x)

  if (a >= 6) return sign
  if (a < 2 ** -28) return sign * ((1 + ERF_COEFF.efx) * a)
  if (a < 0.84375) {
    const z = a * a
    return sign * (a * (1 + polyEval(ERF_COEFF.pp, z) / polyEval(ERF_COEFF.qq, z)))
  }
  if (a < 1.25) {
    const s = a - 1
    return sign * (ERF_COEFF.erx + polyEval(ERF_COEFF.pa, s) / polyEval(ERF_COEFF.qa, s))
  }
  if (a < 1 / 0.35) {
    const z = a * a
    const s = 1 / z
    return sign * (1 - Math.exp(-z - 0.5625 + polyEval(ERF_COEFF.ra, s) / polyEval(ERF_COEFF.sa, s)) / a)
  }
  {
    const z = a * a
    const s = 1 / z
    return sign * (1 - Math.exp(-z - 0.5625 + polyEval(ERF_COEFF.rb, s) / polyEval(ERF_COEFF.sb, s)) / a)
  }
}

function ndtr(x) {
  const v = 0.5 + 0.5 * erfScalar(x / SQRT2)
  if (v <= 0) return 0
  if (v >= 1) return 1
  return v
}

function logNdtr(x) {
  if (x > 6) {
    return -ndtr(-x)
  }
  if (x > -20) {
    return Math.log(ndtr(x))
  }

  const logLhs = -0.5 * x * x - Math.log(-x) - LOG_SQRT_2PI
  let lastTotal = 0
  let rhs = 1
  let numerator = 1
  let denomFactor = 1
  const denomCons = 1 / (x * x)
  let sign = 1
  let i = 0

  while (Math.abs(lastTotal - rhs) > Number.EPSILON) {
    i += 1
    lastTotal = rhs
    sign = -sign
    denomFactor *= denomCons
    numerator *= 2 * i - 1
    rhs += sign * numerator * denomFactor
  }

  return logLhs + Math.log(rhs)
}

function logSum(logP, logQ) {
  const a = Math.max(logP, logQ)
  const b = Math.min(logP, logQ)
  return a + Math.log1p(Math.exp(b - a))
}

function logDiff(logP, logQ) {
  if (logQ >= logP) {
    return -Infinity
  }
  return logP + Math.log1p(-Math.exp(logQ - logP))
}

function logGaussMass(a, b) {
  if (b <= 0) {
    return logDiff(logNdtr(b), logNdtr(a))
  }
  if (a > 0) {
    return logGaussMass(-b, -a)
  }
  const central = 1 - ndtr(a) - ndtr(-b)
  if (central > 0) {
    return Math.log(central)
  }
  // Fallback against cancellation/rounding on boundary values.
  const leftMass = logDiff(logNdtr(0), logNdtr(a))
  const rightMass = logDiff(logNdtr(b), logNdtr(0))
  return logSum(leftMass, rightMass)
}

function ndtriExp(y) {
  let flipped = false
  let z = y
  if (y > -1e-2) {
    flipped = true
    z = Math.log(-Math.expm1(y))
  }

  let x
  if (z < -5) {
    x = -Math.sqrt(-2 * (z + LOG_SQRT_2PI))
  } else {
    x = -NDTRI_EXP_APPROX_C * Math.log(Math.expm1(-z))
  }

  for (let i = 0; i < 100; i += 1) {
    const logNdtrX = logNdtr(x)
    const logNormPdfX = -0.5 * x * x - LOG_SQRT_2PI
    const dx = (logNdtrX - z) * Math.exp(logNdtrX - logNormPdfX)
    x -= dx
    if (Math.abs(dx) < 1e-8 * Math.abs(x)) {
      break
    }
  }

  if (flipped) {
    x *= -1
  }

  return x
}

function truncnormPpf(q, a, b) {
  if (q === 0) return a
  if (q === 1) return b
  if (a === b) return Number.NaN

  const logMass = logGaussMass(a, b)
  if (a < 0) {
    const logPhiX = logSum(logNdtr(a), Math.log(q) + logMass)
    return ndtriExp(logPhiX)
  }

  const logPhiX = logSum(logNdtr(-b), Math.log1p(-q) + logMass)
  return -ndtriExp(logPhiX)
}

function truncnormRvs(a, b, loc, scale, rng) {
  const q = rng.randomSample()
  return truncnormPpf(q, a, b) * scale + loc
}

function truncnormLogpdf(x, a, b, loc, scale) {
  const xn = (x - loc) / scale
  const out = -0.5 * xn * xn - LOG_SQRT_2PI - logGaussMass(a, b) - Math.log(scale)
  if (a === b) return Number.NaN
  if (xn < a || xn > b) return -Infinity
  return out
}

class MixtureOfProductDistribution {
  constructor(weights, distributions) {
    this.weights = weights
    this.distributions = distributions
  }

  sample(rng, batchSize) {
    const activeIndices = rng.choiceWeighted(this.weights, batchSize)
    const ret = Array.from({ length: batchSize }, () =>
      new Array(this.distributions.length).fill(0)
    )

    const discNumerical = []
    const numericalDefs = []
    const numericalColumns = []
    const logNumericalColumns = []
    const lowsNumeric = []
    const highsNumeric = []

    for (let col = 0; col < this.distributions.length; col += 1) {
      const distDef = this.distributions[col]
      const d = distDef.distribution
      if (d.kind === 'categorical') {
        const rndQuantile = rng.rand(batchSize)
        for (let row = 0; row < batchSize; row += 1) {
          const weights = d.weights[activeIndices[row]]
          let cum = 0
          let choice = 0
          for (; choice < weights.length; choice += 1) {
            cum += weights[choice]
            if (choice === weights.length - 1) {
              cum = 1
            }
            if (!(cum < rndQuantile[row])) {
              break
            }
          }
          ret[row][col] = choice
        }
        continue
      }

      numericalDefs.push(d)
      numericalColumns.push(col)

      if (d.kind === 'truncnorm') {
        lowsNumeric.push(d.low)
        highsNumeric.push(d.high)
      } else if (d.kind === 'trunclognorm') {
        lowsNumeric.push(Math.log(d.low))
        highsNumeric.push(Math.log(d.high))
        logNumericalColumns.push(col)
      } else if (d.kind === 'discrete_truncnorm') {
        lowsNumeric.push(d.low - d.step / 2)
        highsNumeric.push(d.high + d.step / 2)
        discNumerical.push({
          column: col,
          low: d.low,
          high: d.high,
          step: d.step
        })
      } else if (d.kind === 'discrete_trunclognorm') {
        lowsNumeric.push(Math.log(d.low - d.step / 2))
        highsNumeric.push(Math.log(d.high + d.step / 2))
        logNumericalColumns.push(col)
        discNumerical.push({
          column: col,
          low: d.low,
          high: d.high,
          step: d.step
        })
      } else {
        throw new Error(`Unknown distribution kind: ${d.kind}`)
      }
    }

    if (numericalDefs.length > 0) {
      const activeMus = numericalDefs.map((d) => activeIndices.map((k) => d.mu[k]))
      const activeSigmas = numericalDefs.map((d) => activeIndices.map((k) => d.sigma[k]))

      // Match NumPy broadcast order in truncnorm.rvs (row-major over (n_numerical, batch_size)).
      for (let i = 0; i < numericalDefs.length; i += 1) {
        for (let row = 0; row < batchSize; row += 1) {
          const mu = activeMus[i][row]
          const sigma = activeSigmas[i][row]
          const a = (lowsNumeric[i] - mu) / sigma
          const b = (highsNumeric[i] - mu) / sigma
          const q = rng.randomSample()
          ret[row][numericalColumns[i]] = truncnormPpf(q, a, b) * sigma + mu
        }
      }

      for (const col of logNumericalColumns) {
        for (let row = 0; row < batchSize; row += 1) {
          ret[row][col] = Math.exp(ret[row][col])
        }
      }

      for (const disc of discNumerical) {
        for (let row = 0; row < batchSize; row += 1) {
          const raw = ret[row][disc.column]
          const rounded = disc.low + roundToNearestEven((raw - disc.low) / disc.step) * disc.step
          ret[row][disc.column] = clip(rounded, disc.low, disc.high)
        }
      }
    }

    const result = {}
    for (let col = 0; col < this.distributions.length; col += 1) {
      result[this.distributions[col].paramName] = ret.map((row) => row[col])
    }
    return result
  }

  logPdf(samplesByParam) {
    const firstParam = this.distributions[0].paramName
    const nSamples = samplesByParam[firstParam].length
    const out = new Array(nSamples)
    const nWeights = this.weights.length

    for (let s = 0; s < nSamples; s += 1) {
      const weightedLogPdf = new Array(nWeights).fill(0)

      for (const distDef of this.distributions) {
        const d = distDef.distribution
        const x = samplesByParam[distDef.paramName][s]

        if (d.kind === 'categorical') {
          const idx = Math.trunc(x)
          for (let k = 0; k < nWeights; k += 1) {
            weightedLogPdf[k] += Math.log(d.weights[k][idx])
          }
          continue
        }

        if (d.kind === 'truncnorm') {
          for (let k = 0; k < nWeights; k += 1) {
            const a = (d.low - d.mu[k]) / d.sigma[k]
            const b = (d.high - d.mu[k]) / d.sigma[k]
            weightedLogPdf[k] += truncnormLogpdf(x, a, b, d.mu[k], d.sigma[k])
          }
          continue
        }

        if (d.kind === 'trunclognorm') {
          const logX = Math.log(x)
          for (let k = 0; k < nWeights; k += 1) {
            const a = (Math.log(d.low) - d.mu[k]) / d.sigma[k]
            const b = (Math.log(d.high) - d.mu[k]) / d.sigma[k]
            weightedLogPdf[k] += truncnormLogpdf(logX, a, b, d.mu[k], d.sigma[k])
          }
          continue
        }

        if (d.kind === 'discrete_truncnorm') {
          for (let k = 0; k < nWeights; k += 1) {
            const lowLogMass = logGaussMass(
              (d.low - d.step / 2 - d.mu[k]) / d.sigma[k],
              (d.high + d.step / 2 - d.mu[k]) / d.sigma[k]
            )
            const xMass = logGaussMass(
              (x - d.step / 2 - d.mu[k]) / d.sigma[k],
              (x + d.step / 2 - d.mu[k]) / d.sigma[k]
            )
            weightedLogPdf[k] += xMass - lowLogMass
          }
          continue
        }

        if (d.kind === 'discrete_trunclognorm') {
          const logXMinus = Math.log(x - d.step / 2)
          const logXPlus = Math.log(x + d.step / 2)
          const logLowMinus = Math.log(d.low - d.step / 2)
          const logHighPlus = Math.log(d.high + d.step / 2)
          for (let k = 0; k < nWeights; k += 1) {
            const lowLogMass = logGaussMass(
              (logLowMinus - d.mu[k]) / d.sigma[k],
              (logHighPlus - d.mu[k]) / d.sigma[k]
            )
            const xMass = logGaussMass(
              (logXMinus - d.mu[k]) / d.sigma[k],
              (logXPlus - d.mu[k]) / d.sigma[k]
            )
            weightedLogPdf[k] += xMass - lowLogMass
          }
          continue
        }

        throw new Error(`Unknown distribution kind: ${d.kind}`)
      }

      for (let k = 0; k < nWeights; k += 1) {
        weightedLogPdf[k] += Math.log(this.weights[k])
      }

      let maxValue = -Infinity
      for (let k = 0; k < nWeights; k += 1) {
        if (weightedLogPdf[k] > maxValue) {
          maxValue = weightedLogPdf[k]
        }
      }
      if (maxValue === -Infinity) {
        maxValue = 0
      }

      let sumExp = 0
      for (let k = 0; k < nWeights; k += 1) {
        sumExp += Math.exp(weightedLogPdf[k] - maxValue)
      }
      out[s] = Math.log(sumExp) + maxValue
    }

    return out
  }
}

function defaultGamma(x) {
  return Math.min(Math.ceil(0.1 * x), 25)
}

function defaultWeights(x) {
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

function callWeightsFunc(weightsFunc, n) {
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

class ParzenEstimator {
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
        const sigma = sigma0Magnitude * Math.pow(Math.max(observations.length, 1), -1 / (Object.keys(this.searchSpace).length + 4)) * (high - low)
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
        return { kind: 'truncnorm', mu: mus, sigma: sigmas, low: searchSpace.low, high: searchSpace.high }
      }
      return { kind: 'trunclognorm', mu: mus, sigma: sigmas, low: searchSpace.low, high: searchSpace.high }
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

function normalizeObjectiveValue(value, direction) {
  if (value === null || value === undefined) {
    return Infinity
  }
  if (direction === StudyDirection.MAXIMIZE) {
    return -value
  }
  return value
}

function isParetoFront2D(uniqueLexsortedLossValues) {
  const n = uniqueLexsortedLossValues.length
  if (n === 0) return []
  const onFront = new Array(n).fill(true)
  let currentMin = uniqueLexsortedLossValues[0][1]
  for (let i = 1; i < n; i += 1) {
    const v = uniqueLexsortedLossValues[i][1]
    onFront[i] = v < currentMin
    if (v < currentMin) {
      currentMin = v
    }
  }
  return onFront
}

function isParetoFrontND(uniqueLexsortedLossValues) {
  const lossValues = uniqueLexsortedLossValues.map((row) => row.slice(1))
  const n = lossValues.length
  const onFront = new Array(n).fill(false)
  let remainingIndices = Array.from({ length: n }, (_, i) => i)

  while (remainingIndices.length > 0) {
    const newNonDominatedIndex = remainingIndices[0]
    onFront[newNonDominatedIndex] = true
    const nextRemaining = []
    for (let r = 0; r < remainingIndices.length; r += 1) {
      const idx = remainingIndices[r]
      let anyLess = false
      for (let k = 0; k < lossValues[idx].length; k += 1) {
        if (lossValues[idx][k] < lossValues[newNonDominatedIndex][k]) {
          anyLess = true
          break
        }
      }
      if (anyLess) {
        nextRemaining.push(idx)
      }
    }
    remainingIndices = nextRemaining
  }

  return onFront
}

function isParetoFrontForUniqueSorted(uniqueLexsortedLossValues) {
  const nObjectives = uniqueLexsortedLossValues[0].length
  if (nObjectives === 1) {
    const onFront = new Array(uniqueLexsortedLossValues.length).fill(false)
    onFront[0] = true
    return onFront
  }
  if (nObjectives === 2) {
    return isParetoFront2D(uniqueLexsortedLossValues)
  }
  return isParetoFrontND(uniqueLexsortedLossValues)
}

function isParetoFront(lossValues, assumeUniqueLexsorted = false) {
  if (lossValues.length === 0) return []
  if (assumeUniqueLexsorted) {
    return isParetoFrontForUniqueSorted(lossValues)
  }
  const { uniqueRows, inverse } = uniqueSortedRowsWithInverse(lossValues)
  const uniqueOnFront = isParetoFrontForUniqueSorted(uniqueRows)
  return inverse.map((i) => uniqueOnFront[i])
}

function calculateNondominationRank(lossValues, nBelow = null) {
  if (lossValues.length === 0 || (nBelow !== null && nBelow <= 0)) {
    return new Array(lossValues.length).fill(0)
  }

  const nObjectives = lossValues[0].length
  if (nObjectives === 1) {
    const values = lossValues.map((row) => row[0])
    const sortedUnique = [...new Set(values.slice().sort((a, b) => a - b))]
    const rankMap = new Map()
    for (let i = 0; i < sortedUnique.length; i += 1) {
      rankMap.set(sortedUnique[i], i)
    }
    return values.map((v) => rankMap.get(v))
  }

  const { uniqueRows, inverse } = uniqueSortedRowsWithInverse(lossValues)
  let remaining = uniqueRows.map((row, i) => ({ row, idx: i }))
  const nUnique = uniqueRows.length
  const clippedNBelow = Math.min(nBelow === null ? nUnique : nBelow, nUnique)
  const ranks = new Array(nUnique).fill(0)
  let rank = 0

  while (nUnique - remaining.length < clippedNBelow) {
    const rows = remaining.map((it) => it.row)
    const onFront = isParetoFront(rows, true)
    const next = []
    for (let i = 0; i < remaining.length; i += 1) {
      if (onFront[i]) {
        ranks[remaining[i].idx] = rank
      } else {
        next.push(remaining[i])
      }
    }
    remaining = next
    rank += 1
  }

  for (const r of remaining) {
    ranks[r.idx] = rank
  }

  return inverse.map((i) => ranks[i])
}

function fastNonDominationRank(lossValues, penalty = null, nBelow = null) {
  if (lossValues.length === 0) {
    return []
  }

  let nBelowLocal = nBelow === null ? lossValues.length : nBelow
  if (nBelowLocal <= 0) {
    throw new Error('nBelow must be positive.')
  }

  if (penalty === null) {
    return calculateNondominationRank(lossValues, nBelowLocal)
  }

  if (penalty.length !== lossValues.length) {
    throw new Error('penalty and lossValues length mismatch.')
  }

  const ranks = new Array(lossValues.length).fill(-1)

  const feasibleIndices = []
  const infeasibleIndices = []
  const nanIndices = []

  for (let i = 0; i < penalty.length; i += 1) {
    const p = penalty[i]
    if (Number.isNaN(p)) {
      nanIndices.push(i)
    } else if (p <= 0) {
      feasibleIndices.push(i)
    } else {
      infeasibleIndices.push(i)
    }
  }

  const feasibleLosses = feasibleIndices.map((i) => lossValues[i])
  const feasibleRanks = calculateNondominationRank(feasibleLosses, nBelowLocal)
  for (let i = 0; i < feasibleIndices.length; i += 1) {
    ranks[feasibleIndices[i]] = feasibleRanks[i]
  }
  nBelowLocal -= feasibleIndices.length

  const topRankInfeasible = Math.max(-1, ...feasibleIndices.map((i) => ranks[i])) + 1
  const infeasiblePenaltyAsLoss = infeasibleIndices.map((i) => [penalty[i]])
  const infeasibleRanks = calculateNondominationRank(infeasiblePenaltyAsLoss, nBelowLocal)
  for (let i = 0; i < infeasibleIndices.length; i += 1) {
    ranks[infeasibleIndices[i]] = topRankInfeasible + infeasibleRanks[i]
  }
  nBelowLocal -= infeasibleIndices.length

  const topRankPenaltyNan =
    Math.max(-1, ...ranks.filter((r, i) => !Number.isNaN(penalty[i]))) + 1
  const nanLosses = nanIndices.map((i) => lossValues[i])
  const nanRanks = calculateNondominationRank(nanLosses, nBelowLocal)
  for (let i = 0; i < nanIndices.length; i += 1) {
    ranks[nanIndices[i]] = topRankPenaltyNan + nanRanks[i]
  }

  return ranks
}

function compute2dHypervolume(sortedParetoSols, referencePoint) {
  let hv = 0
  for (let i = 0; i < sortedParetoSols.length; i += 1) {
    const rectDiagY = i === 0 ? referencePoint[1] : sortedParetoSols[i - 1][1]
    const edgeX = referencePoint[0] - sortedParetoSols[i][0]
    const edgeY = rectDiagY - sortedParetoSols[i][1]
    hv += edgeX * edgeY
  }
  return hv
}

function compute3dHypervolume(sortedParetoSols, referencePoint) {
  const n = sortedParetoSols.length
  const yOrder = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => sortedParetoSols[a][1] - sortedParetoSols[b][1]
  )

  const zDelta = Array.from({ length: n }, () => new Array(n).fill(0))
  for (let j = 0; j < n; j += 1) {
    const row = yOrder[j]
    zDelta[row][j] = referencePoint[2] - sortedParetoSols[row][2]
  }

  for (let i = 0; i < n; i += 1) {
    for (let j = 1; j < n; j += 1) {
      if (zDelta[i][j] < zDelta[i][j - 1]) {
        zDelta[i][j] = zDelta[i][j - 1]
      }
    }
  }
  for (let j = 0; j < n; j += 1) {
    for (let i = 1; i < n; i += 1) {
      if (zDelta[i][j] < zDelta[i - 1][j]) {
        zDelta[i][j] = zDelta[i - 1][j]
      }
    }
  }

  const xVals = sortedParetoSols.map((row) => row[0])
  const yVals = yOrder.map((idx) => sortedParetoSols[idx][1])

  const xDelta = new Array(n)
  const yDelta = new Array(n)
  for (let i = 0; i < n; i += 1) {
    xDelta[i] = (i + 1 < n ? xVals[i + 1] : referencePoint[0]) - xVals[i]
    yDelta[i] = (i + 1 < n ? yVals[i + 1] : referencePoint[1]) - yVals[i]
  }

  let hv = 0
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      hv += zDelta[j][i] * yDelta[i] * xDelta[j]
    }
  }

  return hv
}

function computeHvRecursive(sortedLossVals, referencePoint) {
  const n = sortedLossVals.length
  if (n === 1) {
    let hv = 1
    for (let i = 0; i < referencePoint.length; i += 1) {
      hv *= referencePoint[i] - sortedLossVals[0][i]
    }
    return hv
  }

  if (n === 2) {
    let hv1 = 1
    let hv2 = 1
    let inter = 1
    for (let i = 0; i < referencePoint.length; i += 1) {
      hv1 *= referencePoint[i] - sortedLossVals[0][i]
      hv2 *= referencePoint[i] - sortedLossVals[1][i]
      inter *= referencePoint[i] - Math.max(sortedLossVals[0][i], sortedLossVals[1][i])
    }
    return hv1 + hv2 - inter
  }

  const inclusiveHvs = sortedLossVals.map((row) => {
    let p = 1
    for (let i = 0; i < row.length; i += 1) {
      p *= referencePoint[i] - row[i]
    }
    return p
  })

  const limitedSolsArray = sortedLossVals.map((rowI) =>
    sortedLossVals.map((rowJ) => rowI.map((v, dim) => Math.max(v, rowJ[dim])))
  )

  let hv = inclusiveHvs[inclusiveHvs.length - 1]
  for (let i = 0; i < inclusiveHvs.length - 1; i += 1) {
    const limited = limitedSolsArray[i].slice(i + 1)
    hv += computeExclusiveHv(limited, inclusiveHvs[i], referencePoint)
  }
  return hv
}

function computeExclusiveHv(limitedSols, inclusiveHv, referencePoint) {
  if (limitedSols.length <= 3) {
    return inclusiveHv - computeHvRecursive(limitedSols, referencePoint)
  }
  const onFront = isParetoFront(limitedSols, true)
  const front = []
  for (let i = 0; i < limitedSols.length; i += 1) {
    if (onFront[i]) {
      front.push(limitedSols[i])
    }
  }
  return inclusiveHv - computeHvRecursive(front, referencePoint)
}

function computeHypervolume(lossVals, referencePoint, assumePareto = false) {
  for (let i = 0; i < lossVals.length; i += 1) {
    for (let d = 0; d < referencePoint.length; d += 1) {
      if (lossVals[i][d] > referencePoint[d]) {
        throw new Error('All points must dominate or equal reference point.')
      }
    }
  }
  for (let i = 0; i < referencePoint.length; i += 1) {
    if (!Number.isFinite(referencePoint[i])) {
      return Infinity
    }
  }
  if (lossVals.length === 0) {
    return 0
  }

  let sortedParetoSols
  if (!assumePareto) {
    const { uniqueRows } = uniqueSortedRowsWithInverse(lossVals)
    const onFront = isParetoFront(uniqueRows, true)
    sortedParetoSols = uniqueRows.filter((_, i) => onFront[i])
  } else {
    sortedParetoSols = [...lossVals].sort((a, b) => a[0] - b[0])
  }

  let hv
  if (referencePoint.length === 2) {
    hv = compute2dHypervolume(sortedParetoSols, referencePoint)
  } else if (referencePoint.length === 3) {
    hv = compute3dHypervolume(sortedParetoSols, referencePoint)
  } else {
    hv = computeHvRecursive(sortedParetoSols, referencePoint)
  }

  return Number.isFinite(hv) ? hv : Infinity
}

function solveHssp2d(rankLossVals, rankIndices, subsetSize, referencePoint) {
  const nTrials = rankLossVals.length
  let sortedIndices = Array.from({ length: nTrials }, (_, i) => i)
  let sortedLossVals = rankLossVals.map((row) => row.slice())
  let rectDiags = Array.from({ length: nTrials }, () => referencePoint.slice())
  const selectedIndices = new Array(subsetSize)

  for (let i = 0; i < subsetSize; i += 1) {
    let maxIndex = 0
    let maxContrib = -Infinity
    for (let j = 0; j < sortedLossVals.length; j += 1) {
      const contrib = (rectDiags[j][0] - sortedLossVals[j][0]) * (rectDiags[j][1] - sortedLossVals[j][1])
      if (contrib > maxContrib) {
        maxContrib = contrib
        maxIndex = j
      }
    }

    selectedIndices[i] = rankIndices[sortedIndices[maxIndex]]
    const loss = sortedLossVals[maxIndex].slice()

    const keep = []
    for (let j = 0; j < sortedLossVals.length; j += 1) {
      if (j !== maxIndex) keep.push(j)
    }

    const nextSortedIndices = []
    const nextRectDiags = []
    const nextLossVals = []
    for (let j = 0; j < keep.length; j += 1) {
      const source = keep[j]
      nextSortedIndices.push(sortedIndices[source])
      nextRectDiags.push(rectDiags[source].slice())
      nextLossVals.push(sortedLossVals[source].slice())
    }

    for (let j = 0; j < nextRectDiags.length; j += 1) {
      if (j < maxIndex) {
        nextRectDiags[j][0] = Math.min(loss[0], nextRectDiags[j][0])
      } else {
        nextRectDiags[j][1] = Math.min(loss[1], nextRectDiags[j][1])
      }
    }

    sortedIndices = nextSortedIndices
    rectDiags = nextRectDiags
    sortedLossVals = nextLossVals
  }

  return selectedIndices
}

function lazyContribsUpdate(contribs, paretoLossValues, selectedVecs, referencePoint, hvSelected) {
  if (!Number.isFinite(hvSelected)) {
    return new Array(contribs.length).fill(Infinity)
  }

  const inclusiveHvs = paretoLossValues.map((row) => {
    let p = 1
    for (let i = 0; i < row.length; i += 1) {
      p *= referencePoint[i] - row[i]
    }
    return p
  })

  const intersec = paretoLossValues.map((row) => selectedVecs.slice(0, -1).map((sel) => row.map((v, d) => Math.max(v, sel[d]))))

  let updated = [...contribs]
  for (let i = 0; i < updated.length; i += 1) {
    const latestIntersection = paretoLossValues[i].map((v, d) => Math.max(v, selectedVecs[selectedVecs.length - 1][d]))
    let latestVolume = 1
    for (let d = 0; d < latestIntersection.length; d += 1) {
      latestVolume *= referencePoint[d] - latestIntersection[d]
    }
    updated[i] = Math.min(updated[i], inclusiveHvs[i] - latestVolume)
  }

  let maxContrib = 0
  const isHvCalcFast = paretoLossValues[0].length <= 3
  const order = Array.from({ length: updated.length }, (_, i) => i).sort((a, b) => updated[b] - updated[a])

  for (const i of order) {
    if (!Number.isFinite(inclusiveHvs[i])) {
      maxContrib = updated[i] = Infinity
      continue
    }
    if (updated[i] < maxContrib) {
      continue
    }

    if (isHvCalcFast) {
      const plusSet = [...selectedVecs]
      plusSet[plusSet.length - 1] = paretoLossValues[i].slice()
      const hvPlus = computeHypervolume(plusSet, referencePoint, true)
      updated[i] = hvPlus - hvSelected
    } else {
      updated[i] = inclusiveHvs[i] - computeHypervolume(intersec[i], referencePoint)
    }

    if (updated[i] > maxContrib) {
      maxContrib = updated[i]
    }
  }

  return updated
}

function solveHsspOnUniqueLossVals(rankLossVals, rankIndices, subsetSize, referencePoint) {
  if (!referencePoint.every((v) => Number.isFinite(v))) {
    return rankIndices.slice(0, subsetSize)
  }
  if (rankIndices.length === subsetSize) {
    return rankIndices.slice()
  }
  if (rankLossVals[0].length === 2) {
    return solveHssp2d(rankLossVals, rankIndices, subsetSize, referencePoint)
  }

  let contribs = rankLossVals.map((row) => {
    let p = 1
    for (let i = 0; i < row.length; i += 1) {
      p *= referencePoint[i] - row[i]
    }
    return p
  })
  let selectedIndices = []
  let selectedVecs = []
  let indices = Array.from({ length: rankLossVals.length }, (_, i) => i)
  let lossVals = rankLossVals.map((row) => row.slice())
  let hv = 0

  for (let k = 0; k < subsetSize; k += 1) {
    let maxIndex = 0
    for (let i = 1; i < contribs.length; i += 1) {
      if (contribs[i] > contribs[maxIndex]) {
        maxIndex = i
      }
    }

    hv += contribs[maxIndex]
    selectedIndices.push(indices[maxIndex])
    selectedVecs.push(lossVals[maxIndex].slice())

    if (k === subsetSize - 1) {
      break
    }

    const keep = []
    for (let i = 0; i < contribs.length; i += 1) {
      if (i !== maxIndex) keep.push(i)
    }

    contribs = keep.map((i) => contribs[i])
    indices = keep.map((i) => indices[i])
    lossVals = keep.map((i) => lossVals[i])

    const selectedForUpdate = [...selectedVecs, new Array(referencePoint.length).fill(0)]
    contribs = lazyContribsUpdate(contribs, lossVals, selectedForUpdate, referencePoint, hv)
  }

  return selectedIndices.map((i) => rankIndices[i])
}

function solveHssp(rankLossVals, rankIndices, subsetSize, referencePoint) {
  if (subsetSize === rankIndices.length) {
    return rankIndices.slice()
  }

  const { uniqueRows, inverse } = uniqueSortedRowsWithInverse(rankLossVals)
  const firstOccurrence = new Array(uniqueRows.length).fill(-1)
  for (let i = 0; i < inverse.length; i += 1) {
    const u = inverse[i]
    if (firstOccurrence[u] === -1) {
      firstOccurrence[u] = i
    }
  }

  if (uniqueRows.length < subsetSize) {
    const chosen = new Array(rankIndices.length).fill(false)
    for (const idx of firstOccurrence) chosen[idx] = true
    const duplicated = []
    for (let i = 0; i < chosen.length; i += 1) {
      if (!chosen[i]) duplicated.push(i)
    }
    for (let i = 0; i < subsetSize - uniqueRows.length; i += 1) {
      chosen[duplicated[i]] = true
    }
    const out = []
    for (let i = 0; i < chosen.length; i += 1) {
      if (chosen[i]) out.push(rankIndices[i])
    }
    return out
  }

  const selectedUnique = solveHsspOnUniqueLossVals(
    uniqueRows,
    firstOccurrence,
    subsetSize,
    referencePoint
  )
  return selectedUnique.map((i) => rankIndices[i])
}

function getReferencePoint(lossVals) {
  const dims = lossVals[0].length
  const worst = new Array(dims).fill(-Infinity)
  for (const row of lossVals) {
    for (let d = 0; d < dims; d += 1) {
      if (row[d] > worst[d]) worst[d] = row[d]
    }
  }

  const ref = worst.map((w) => Math.max(1.1 * w, 0.9 * w))
  for (let i = 0; i < ref.length; i += 1) {
    if (ref[i] === 0) ref[i] = EPS
  }
  return ref
}

function splitCompleteTrialsSingleObjective(trials, study, nBelow) {
  const sorted = [...trials].sort((a, b) => {
    if (study.direction === StudyDirection.MINIMIZE) {
      return a.value - b.value
    }
    return b.value - a.value
  })
  return [sorted.slice(0, nBelow), sorted.slice(nBelow)]
}

function splitCompleteTrialsMultiObjective(trials, study, nBelow) {
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
    const selected = solveHssp(rankLossVals, needIndices, subsetSize, getReferencePoint(rankLossVals))
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

function splitCompleteTrials(trials, study, nBelow) {
  const clipped = Math.min(nBelow, trials.length)
  if (study.directions.length <= 1) {
    return splitCompleteTrialsSingleObjective(trials, study, clipped)
  }
  return splitCompleteTrialsMultiObjective(trials, study, clipped)
}

function getPrunedTrialScore(trial, study) {
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

function splitPrunedTrials(trials, study, nBelow) {
  const clipped = Math.min(nBelow, trials.length)
  const sorted = [...trials].sort((a, b) => {
    const sa = getPrunedTrialScore(a, study)
    const sb = getPrunedTrialScore(b, study)
    if (sa[0] !== sb[0]) return sa[0] - sb[0]
    return sa[1] - sb[1]
  })
  return [sorted.slice(0, clipped), sorted.slice(clipped)]
}

function getInfeasibleTrialScore(trial) {
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

function splitInfeasibleTrials(trials, nBelow) {
  const clipped = Math.min(nBelow, trials.length)
  const sorted = [...trials].sort((a, b) => getInfeasibleTrialScore(a) - getInfeasibleTrialScore(b))
  return [sorted.slice(0, clipped), sorted.slice(clipped)]
}

function splitTrials(study, trials, nBelow, constraintsEnabled) {
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

  const below = [...belowComplete, ...belowPruned, ...belowInfeasible].sort((a, b) => a.number - b.number)
  const above = [...aboveComplete, ...abovePruned, ...aboveInfeasible, ...running].sort((a, b) => a.number - b.number)

  return [below, above]
}

function calculateWeightsBelowForMultiObjective(study, belowTrials, constraintsFunc) {
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

function processConstraintsAfterTrial(constraintsFunc, study, trial, state) {
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

class TPESampler {
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
        trial.system_attrs[`tpe:relative_params:${Math.floor(i / maxLen)}`] = paramsStr.slice(i, i + maxLen)
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
      const weightsBelow = calculateWeightsBelowForMultiObjective(study, trials, this.constraintsFunc)
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
              const minDomainValue =
                dist instanceof CategoricalDistribution ? 0 : dist.low

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

class TrialRuntime {
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
    this.relativeParams = sampler.sampleRelative(this.study, this.frozen, this.relativeSearchSpace)
    this.relativePrepared = true
  }

  _suggest(name, distribution) {
    if (name in this.frozen.params) {
      return this.frozen.params[name]
    }

    this.ensureRelativePrepared()
    this.frozen.distributions[name] = distribution

    let value
    if (name in this.relativeSearchSpace && name in this.relativeParams) {
      value = this.relativeParams[name]
    } else {
      value = this.study.sampler.sampleIndependent(this.study, this.frozen, name, distribution)
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

export class Study {
  constructor({ sampler, directions }) {
    this.sampler = sampler
    this.directions = directions
    this.direction = directions[0]
    this.trials = []
  }

  isMultiObjective() {
    return this.directions.length > 1
  }

  ask() {
    const number = this.trials.length
    const frozen = {
      number,
      state: TrialState.RUNNING,
      params: {},
      distributions: {},
      system_attrs: {},
      intermediate_values: {},
      value: null,
      values: null
    }

    this.trials.push(frozen)
    this.sampler.beforeTrial(this, frozen)
    return new TrialRuntime(this, frozen)
  }

  tell(trialRuntime, { value = null, values = null, state = null } = {}) {
    const frozen = trialRuntime instanceof TrialRuntime ? trialRuntime.frozen : trialRuntime
    if (state === null || state === undefined) {
      state = TrialState.COMPLETE
    }

    if (state === TrialState.COMPLETE) {
      if (this.isMultiObjective()) {
        if (!Array.isArray(values)) {
          throw new Error('Multi-objective study requires values array on complete trial.')
        }
        frozen.values = values.slice()
        frozen.value = null
      } else {
        if (typeof value !== 'number') {
          throw new Error('Single-objective study requires numeric value on complete trial.')
        }
        frozen.value = value
        frozen.values = null
      }
    }

    frozen.state = state
    this.sampler.afterTrial(this, frozen, state, frozen.values)
  }

  getTrials({ states = null, useCache = true } = {}) {
    void useCache
    if (states === null) {
      return this.trials
    }
    return this.trials.filter((trial) => states.includes(trial.state))
  }

  serialize() {
    return {
      magic: STUDY_SNAPSHOT_MAGIC,
      version: STUDY_SNAPSHOT_VERSION,
      directions: this.directions.slice(),
      sampler: serializeSamplerForSnapshot(this.sampler),
      trials: this.trials.map((trial) => serializeTrialForSnapshot(trial))
    }
  }

  toJSON() {
    return this.serialize()
  }

  static deserialize(snapshot, options = {}) {
    if (!isPlainObject(snapshot)) {
      throw new Error('Invalid study snapshot: expected an object.')
    }
    if (snapshot.magic !== STUDY_SNAPSHOT_MAGIC) {
      throw new Error(
        `Invalid study snapshot magic "${snapshot.magic}". Expected "${STUDY_SNAPSHOT_MAGIC}".`
      )
    }
    if (snapshot.version !== STUDY_SNAPSHOT_VERSION) {
      throw new Error(
        `Unsupported study snapshot version ${snapshot.version}. Expected ${STUDY_SNAPSHOT_VERSION}.`
      )
    }
    if (!Array.isArray(snapshot.directions) || snapshot.directions.length === 0) {
      throw new Error('Invalid study snapshot: missing directions.')
    }
    if (!Array.isArray(snapshot.trials)) {
      throw new Error('Invalid study snapshot: missing trials array.')
    }

    const sampler = deserializeSamplerFromSnapshot(snapshot.sampler, options)
    const study = new Study({
      sampler,
      directions: snapshot.directions.slice()
    })
    study.trials = snapshot.trials.map((trial) => deserializeTrialFromSnapshot(trial))
    return study
  }

  static fromJSON(snapshot, options = {}) {
    return Study.deserialize(snapshot, options)
  }

  static parse(serializedJson, options = {}) {
    return Study.deserialize(JSON.parse(serializedJson), options)
  }
}

export function createTPESampler(options = {}) {
  return new TPESampler(options)
}

export function serializeStudy(study) {
  if (!(study instanceof Study)) {
    throw new Error('serializeStudy expects an instance of Study.')
  }
  return study.serialize()
}

export function deserializeStudy(snapshot, options = {}) {
  return Study.deserialize(snapshot, options)
}

export function sanitizeParams(params) {
  const out = {}
  for (const [key, value] of Object.entries(params)) {
    if (typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean') {
      out[key] = value
    } else {
      out[key] = String(value)
    }
  }
  return out
}

// Debug/testing export: useful to inspect parity with Optuna internals.
export { splitTrials }
