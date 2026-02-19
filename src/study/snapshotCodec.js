import {
  FUNCTION_SPEC_KIND
} from '../core/constants.js'
import { isPlainObject } from '../core/objectUtils.js'
import {
  deserializeJsonValueFromSnapshot,
  serializeJsonValueForSnapshot
} from '../core/snapshotJson.js'
import {
  CategoricalDistribution,
  FloatDistribution,
  IntDistribution
} from '../distributions/distributions.js'
import { defaultGamma, defaultWeights } from '../parzen/parzenEstimator.js'
import { TPESampler } from '../sampler/tpeSampler.js'

export function serializeDistributionForSnapshot(distribution) {
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

export function deserializeDistributionFromSnapshot(payload) {
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

export function serializeRngStateForSnapshot(rng) {
  return {
    mt: Array.from(rng.mt),
    mti: rng.mti
  }
}

export function restoreRngStateFromSnapshot(rng, state) {
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

export function serializeFunctionSpec(fn, builtinFn, builtinName) {
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

export function resolveFunctionSpec(
  spec,
  builtinFn,
  builtinName,
  overrideFn,
  label
) {
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

export function serializeOptionalFunctionSpec(fn) {
  if (fn === null || fn === undefined) {
    return { kind: FUNCTION_SPEC_KIND.NONE }
  }
  return {
    kind: FUNCTION_SPEC_KIND.CUSTOM,
    name: typeof fn === 'function' && fn.name ? fn.name : null
  }
}

export function resolveOptionalFunctionSpec(spec, overrideFn, label) {
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

export function serializeCategoricalDistanceSpec(distanceFuncMap) {
  if (!distanceFuncMap || Object.keys(distanceFuncMap).length === 0) {
    return { kind: FUNCTION_SPEC_KIND.NONE }
  }
  const keys = Object.keys(distanceFuncMap).sort()
  return {
    kind: FUNCTION_SPEC_KIND.CUSTOM,
    keys
  }
}

export function resolveCategoricalDistanceSpec(spec, overrideMap) {
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
  throw new Error(
    `Unsupported categorical distance function spec kind "${spec.kind}".`
  )
}

export function serializeSamplerForSnapshot(sampler) {
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

export function deserializeSamplerFromSnapshot(payload, options = {}) {
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
    gamma: resolveFunctionSpec(
      config.gamma,
      defaultGamma,
      'defaultGamma',
      functions.gamma,
      'gamma'
    ),
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

export function serializeTrialForSnapshot(trial) {
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

export function deserializeTrialFromSnapshot(payload) {
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
